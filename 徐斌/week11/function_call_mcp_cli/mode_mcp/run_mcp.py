"""
run_mcp.py — 方式二：MCP Host（连接 Server，多轮闭环调用）

教学重点：
  1. 工具来自"协议发现"而非手写：connect_all_servers 一次走完
     stdio_client 建管道 → initialize() 握手 → list_tools() 发现工具
  2. MCP 工具描述要转成 LLM 能懂的 OpenAI tools schema（inputSchema → parameters）
  3. 多轮闭环代码和 run_function_call.py 几乎一样——差异只在"工具从哪来/怎么执行"
     · Function Call：手写 schema + 直接调后端函数
     · MCP：发现 schema + 通过 call_tool 跨进程调用 Server
  4. 天气：geocode_city → get_weather_by_coords，依赖前一步结果的串行循环
  5. AsyncExitStack 统一管理 Server 子进程生命周期

使用方式：
  python mode_mcp/run_mcp.py --question "宁德现在天气怎么样？"
  python mode_mcp/run_mcp.py --demo

依赖：
  pip install mcp openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM）

MCP 三角关系：
  Host（本文件）= 连接管理 + 工具路由 + LLM 多轮闭环
  Client        = ClientSession
  Server        = weather_server.py（子进程，stdio 通信）
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


def build_server_configs() -> dict[str, StdioServerParameters]:
    servers = BASE_DIR / "mode_mcp" / "servers"
    return {
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }


async def connect_all_servers(stack: AsyncExitStack):
    """
    连接 MCP Server，返回 (tool_registry, openai_tools)：
      tool_registry : tool_name → (ClientSession, server_label)
      openai_tools  : 转成 OpenAI tools schema 的列表
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


MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = (
    "你是一名天气助手。回答天气相关问题时必须分两步调用工具："
    "先调用 geocode_city 获取坐标，再根据返回的 latitude/longitude 调用 get_weather_by_coords；"
    "不要臆造坐标或天气数据，只依据工具返回结果作答。"
    "若某工具依赖另一步的结果，请先调前者，等结果回填后再调后者。"
)


async def run(client, model: str, question: str,
              tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

        if verbose:
            print(f"  —— 工具轮次 {round_idx} ——")
        messages.append(msg)

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "round": round_idx})
            if verbose:
                print(f"  → [mcp] {name}({args})")

            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                call_result = await session.call_tool(name, args)
                result = "\n".join(b.text for b in call_result.content if hasattr(b, "text"))

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    else:
        if verbose:
            print(f"  ⚠ 已达工具轮次上限 {MAX_TOOL_ROUNDS}，强制生成最终回答")
        resp = client.chat.completions.create(model=model, messages=messages)
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，工具轮次日志 {len(tool_call_log)} 次调用）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


DEMO_QUESTIONS = [
    "宁德现在天气怎么样？",
    "北京今天天气如何？另外未来三天呢？",
    "对比一下上海和深圳现在的天气。",
]


async def main_async(provider: str, question: str | None, demo: bool, verbose: bool, as_json: bool):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP] provider={provider} model={model}\n", file=sys.stderr)

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
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args.provider, args.question, args.demo, verbose=not args.quiet, as_json=args.json))


if __name__ == "__main__":
    main()
