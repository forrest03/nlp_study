"""Function Call 循环调用版；基础单轮教学代码位于 mode_function_call/。"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mode_function_call.run_function_call import (  # noqa: E402
    DEMO_QUESTIONS,
    PROVIDERS,
    SYSTEM_PROMPT as BASE_SYSTEM_PROMPT,
    TOOLS_SCHEMA,
    TOOL_DISPATCH,
    build_client,
)

DEFAULT_MAX_TOOL_ROUNDS = 5
SYSTEM_PROMPT = (
    BASE_SYSTEM_PROMPT
    + "你可以一次调用多个工具，也可以根据上一轮工具结果继续调用。"
)


def run(client, model: str, question: str, verbose: bool = True,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS) -> dict:
    """循环执行并回填工具调用，直到模型回答或达到最大轮数。"""
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
            tools=TOOLS_SCHEMA,
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
                print(f"  → [tool/loop] {name}({args})")

            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="none",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm/loop] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Function Call 循环调用扩展")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-tool-rounds", type=int, default=DEFAULT_MAX_TOOL_ROUNDS,
                        help="单个问题允许的最大工具调用轮数（默认 5）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call/Loop] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, question in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{question}")
            print("=" * 60)
        result = run(
            client,
            model,
            question,
            verbose=not (args.quiet or args.json),
            max_tool_rounds=args.max_tool_rounds,
        )
        result["question"] = question
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
