"""
run_mcp_loop.py — 方式二（循环版）：MCP Host + 多轮 ReAct 循环

教学重点：
  1. 单轮闭环的局限：run_mcp.py 只允许"模型调一次工具 → 看结果 → 出答案"，
     一旦模型想"先查年报、再根据结果决定要不要换个查询再查一次"，就做不到——
     ARCHITECTURE.md 第 99 行点名了这是 Agent 多轮循环要解决的问题。
  2. 循环调用的核心：把 create → tool_calls → 路由到 Server 执行 → 回填 → 再 create 包成 while 循环，
     每轮模型自己决定"继续调工具"还是"给出最终答案"（无 tool_calls 即退出循环）。
  3. MCP 特性保留：仍走 stdio JSON-RPC 跨进程调用 Server，AsyncExitStack 管理生命周期，
     只是把"协议层"从单轮改成多轮，Server 代码零改动。
  4. 安全护栏：MAX_ITER 防止死循环——Agent 循环必备兜底。

与原版 run_mcp.run() 的唯一差异（仅 protocol 层）：
  · run_mcp.run()        ：if msg.tool_calls → 路由执行 → 再 create 一次 → 结束
  · 本文件 run()         ：while msg.tool_calls → 路由执行 → 再 create → 直到无 tool_calls

使用方式：
  python mode_mcp_loop/run_mcp_loop.py --question "宁德时代总部宁德的天气如何？如果下雨，推荐一个南方城市并查它的天气"
  python mode_mcp_loop/run_mcp_loop.py --demo

依赖：
  pip install mcp openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM）
            DASHSCOPE_API_KEY（Embedding，rag_server 内部用）
"""

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

# ── LLM 配置（与 run_mcp.py 完全一致，便于横向对比）──────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── Server 配置（与 run_mcp.py 一致，复用现有两个 Server 子进程）──────────

def build_server_configs() -> dict[str, StdioServerParameters]:
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "rag": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "rag_server.py")],
            env={**os.environ},
        ),
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


async def connect_all_servers(stack: AsyncExitStack):
    """
    连接所有 MCP Server，返回 (tool_registry, openai_tools)：
      tool_registry : tool_name → (ClientSession, server_label)，用于路由 call_tool
      openai_tools  : 转成 OpenAI tools schema 的列表，直接喂给 LLM
    """
    print("正在连接 MCP Servers...\n", file=sys.stderr)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    openai_tools: list[dict] = []

    for label, params in build_server_configs().items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_registry[tool.name] = (session, label)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        print(f"  ✓ [{label}]  {', '.join(t.name for t in tools_result.tools)}", file=sys.stderr)

    print(f"\n共 {len(tool_registry)} 个工具就绪\n", file=sys.stderr)
    return tool_registry, openai_tools


# ── 多轮循环 ───────────────────────────────────────────────────────────────

MAX_ITER = 10

SYSTEM_PROMPT = (
    "你是一名金融分析助手，具备多轮工具调用能力。"
    "回答 A 股年报问题时必须先调用 search_annual_report 工具检索年报原文，只依据工具返回的段落作答，不要编造。"
    "知识库仅含 5 家公司：贵州茅台/五粮液/宁德时代/海康威视/中国平安，年份 2021-2023；不在库内的请明确告知。"
    "涉及天气时调用 get_weather。"
    "你可以分多轮调用工具：先查一个城市/公司，根据结果再决定是否继续查询其他城市/公司。"
    "一旦信息足够回答用户问题，请直接给出最终答案，不要再调用工具。"
)


async def _execute_tool_call(tc, tool_registry: dict, verbose: bool, iteration: int) -> tuple[str, dict]:
    """路由单个 tool_call 到对应 Server 执行，返回 (结果字符串, 日志条目)。"""
    name = tc.function.name
    args = json.loads(tc.function.arguments or "{}")
    log_entry = {"name": name, "args": args, "round": iteration}
    if verbose:
        print(f"  → [round {iteration}] [mcp] {name}({args})")

    session, label = tool_registry.get(name, (None, None))
    if session is None:
        result = f"未知工具：{name}"
    else:
        # call_tool() = MCP 协议的 tools/call 请求，工具在 Server 子进程内执行
        call_result = await session.call_tool(name, args)
        result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))

    preview = (result or "")[:120].replace("\n", " ")
    if verbose:
        print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
    return result, log_entry


async def run(client, model: str, question: str,
              tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> dict:
    """
    多轮循环：提问 → 循环{ 模型输出 tool_call → 路由到 Server 执行 → 回填 → 再请求 } → 最终回答。
    与 run_mcp.run() 的唯一差别：把"if msg.tool_calls"改成"while msg.tool_calls"。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    iteration = 0
    msg = None

    # 【教学时刻 1】：while 循环是本文件的核心——模型不再只能调一次工具
    while iteration < MAX_ITER:
        iteration += 1
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型本轮没有 tool_calls = 已生成最终答案，退出循环
        if not msg.tool_calls:
            if verbose:
                print(f"  → [round {iteration}] 模型未调用工具，直接作答\n")
            break

        # 有 tool_calls = 路由到对应 Server 执行后回填，进入下一轮
        messages.append(msg)
        for tc in msg.tool_calls:
            result, log_entry = await _execute_tool_call(tc, tool_registry, verbose, iteration)
            tool_call_log.append(log_entry)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        # 循环回到 while 顶部：再次请求模型，它可以继续调工具或给出答案
    else:
        # while 正常走完 MAX_ITER 次仍没退出——强制再请求一次拿最终答案
        if verbose:
            print(f"  → [!] 达到 MAX_ITER={MAX_ITER}，强制收尾\n")
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="none",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，共 {iteration} 轮，{len(tool_call_log)} 次工具调用）")
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "iterations": iteration,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代总部宁德的天气如何？如果下雨，推荐一个南方城市并查它的天气。",
    "帮我依次查北京、上海、广州的天气，并告诉我哪座城市最适合明天出门。",
    "宁德时代的总部在哪？查一下总部所在地的天气。",
    "对比贵州茅台和五粮液2023年的营收，如果数据不够清楚可以再查一次。",
]


async def main_async(provider: str, question: str | None, demo: bool, verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP Loop] provider={provider} model={model} max_iter={MAX_ITER}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)

        questions = DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
        results = []
        for i, q in enumerate(questions, 1):
            if not as_json:
                print("=" * 60)
                print(f"Q{i}：{q}")
                print("=" * 60)
            result = await run(client, model, q, tool_registry, openai_tools,
                               verbose=verbose and not as_json)
            result["question"] = q
            results.append(result)
            if not as_json:
                print("\n最终回答：")
                print(result["answer"])
                print()

        if as_json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


def main():
    global MAX_ITER
    import argparse
    parser = argparse.ArgumentParser(description="方式二（循环版）：MCP 多轮 ReAct")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-iter", type=int, default=MAX_ITER, help=f"循环上限（默认 {MAX_ITER}）")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    MAX_ITER = args.max_iter
    asyncio.run(main_async(args.provider, args.question, args.demo,
                           verbose=not args.quiet, as_json=args.json))


if __name__ == "__main__":
    main()
