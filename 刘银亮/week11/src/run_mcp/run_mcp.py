import sys
import os
import json
import time
import asyncio
from pathlib import Path
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent

ALIYUN_API_URL  = os.getenv("ALIYUN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL       = "qwen3.7-plus"

def get_client() -> OpenAI:
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 ALIYUN_API_KEY")
    return OpenAI(api_key=api_key, base_url=ALIYUN_API_URL)


# ── Server 配置 ────────────────────────────────────────────────────────────

def build_server_configs() -> dict[str, StdioServerParameters]:
    # 两个自写 Server，都用项目内 Python 脚本启动，stdio 通信
    servers = BASE_DIR / "run_mcp" / "servers"
    return {
        "weather": StdioServerParameters(
            command=sys.executable,
            args=[str(servers / "weather_server.py")],
            env={**os.environ},
        ),
    }

# ── 连接所有 Server：一次走完 建管道→握手→发现工具→转 schema ───────────────
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
        # stdio_client 建立进程间通信管道（子进程的 stdin/stdout）
        read, write = await stack.enter_async_context(stdio_client(params))
        session: ClientSession = await stack.enter_async_context(ClientSession(read, write))

        # initialize() = MCP 握手，协商协议版本和能力
        await session.initialize()

        # list_tools() = 工具发现；同时把 MCP inputSchema 适配成 OpenAI parameters
        # —— 这一步是"协议层 → 模型层"的转换：MCP 让工具与模型解耦，
        #   但喂给具体 LLM 时仍要变成它认识的格式（inputSchema 本就是 JSON Schema，直接塞）
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


SYSTEM_PROMPT = (
    "你是一名旅游博主, 熟悉旅游相关知识以及地理信息。"
    "涉及天气时可以调用 get_weather_data, get_geo 函数。你可以一次调用多个工具。"
)

async def run(client, model: str, question: str,
              tool_registry: dict, openai_tools: list[dict], verbose: bool = True) -> dict:
    """多轮闭环：提问 → 模型输出 tool_call → 路由到 Server 执行 → 回填 → 继续调用直到无工具调用 → 最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    while True:
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools, tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
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

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


async def main_async(question: str):
    client = get_client()

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)

        print("=" * 60)
        print(f"Q：{question}")
        print("=" * 60)
        result = await run(client, LLM_MODEL, question, tool_registry, openai_tools)
        result["question"] = question
        print("\n最终回答：")
        print(result["answer"])
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="方式二：MCP")
    parser.add_argument("--question", "-q", required=True, help="提问内容")
    args = parser.parse_args()
    asyncio.run(main_async(args.question))