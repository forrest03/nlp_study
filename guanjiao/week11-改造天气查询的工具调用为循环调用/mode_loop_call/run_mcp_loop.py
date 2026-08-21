"""MCP 循环调用版；基础单轮教学代码位于 mode_mcp/。"""

import argparse
import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mode_mcp.run_mcp import (  # noqa: E402
    DEMO_QUESTIONS,
    PROVIDERS,
    SYSTEM_PROMPT as BASE_SYSTEM_PROMPT,
    build_client,
    connect_all_servers,
)

DEFAULT_MAX_TOOL_ROUNDS = 5
SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + "你可以一次调用多个工具，也可以根据上一轮工具结果继续调用。"
)


async def run(client, model: str, question: str, tool_registry: dict,
              openai_tools: list[dict], verbose: bool = True,
              max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS) -> dict:
    """循环执行 MCP tool_calls，直到模型回答或达到最大轮数。"""
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds 必须大于等于 1")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
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
                print(f"  → [mcp/loop] {name}({args})")

            session, label = tool_registry.get(name, (None, None))
            if session is None:
                result = f"未知工具：{name}"
            else:
                call_result = await session.call_tool(name, args)
                result = "\n".join(
                    block.text for block in call_result.content if hasattr(block, "text")
                )

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ [{label}] {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="none",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm/loop] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


async def main_async(provider: str, question: str | None, demo: bool, verbose: bool,
                     as_json: bool, max_tool_rounds: int):
    client, model = build_client(provider)
    if not as_json:
        print(f"[MCP/Loop] provider={provider} model={model}\n", file=sys.stderr)

    async with AsyncExitStack() as stack:
        tool_registry, openai_tools = await connect_all_servers(stack)
        questions = DEMO_QUESTIONS if demo else ([question] if question else [DEMO_QUESTIONS[0]])
        results = []
        for i, current_question in enumerate(questions, 1):
            if not as_json:
                print("=" * 60)
                print(f"Q{i}：{current_question}")
                print("=" * 60)
            result = await run(
                client,
                model,
                current_question,
                tool_registry,
                openai_tools,
                verbose=verbose and not as_json,
                max_tool_rounds=max_tool_rounds,
            )
            result["question"] = current_question
            results.append(result)
            if not as_json:
                print("\n最终回答：")
                print(result["answer"])
                print()

        if as_json:
            print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="MCP 循环调用扩展")
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-tool-rounds", type=int, default=DEFAULT_MAX_TOOL_ROUNDS,
                        help="单个问题允许的最大工具调用轮数（默认 5）")
    args = parser.parse_args()
    asyncio.run(main_async(
        args.provider,
        args.question,
        args.demo,
        verbose=not args.quiet,
        as_json=args.json,
        max_tool_rounds=args.max_tool_rounds,
    ))


if __name__ == "__main__":
    main()
