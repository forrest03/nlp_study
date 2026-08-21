import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.region_backend import get_region
from src.weather_backend import get_weather

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": "sk-bbad0654d3374b13903515bb74b04600",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": "sk-ws-H.EMIILEM.re1g.MEUCIHySxOkNu8AKveLvomkwOC5Ri212gohBlrjbXYJzEu33AiEA-_Aw7BSJwouLSuRPj6mVH6--kdcQFzid9O1_hAH7m_w",
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


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_region",
            "description": "查询指定城市的经纬度。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    }
]

TOOL_DISPATCH = {
    "get_region": get_region,
    "get_weather": get_weather,
}

# ── 链式调用 ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一个专业问答助手，请回答用户问题。"
    "涉及城市时调用 get_region。涉及天气时调用 get_weather。"
    "你可以一次调用多个工具。"
)


def run(client, model: str, question: str, verbose: bool = True, max_iterations: int = 5) -> dict:
    """
    链式调用：完全按照伪代码实现
    伪代码：
        resp = model.chat(msgs, tools)
        while resp.has_tool_call():
            result = run(resp.tool_call)
            msgs = append(result)
            resp = model.chat(msgs, tools)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    iteration = 0

    # ====== 第一次请求：resp = model.chat(msgs, tools) ======
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # ====== while resp.has_tool_call(): ======
    while msg.tool_calls and iteration < max_iterations:
        iteration += 1

        # ====== msgs = append(msg) ======
        messages.append(msg)

        # ====== result = run(resp.tool_call) ======
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "iteration": iteration})

            if verbose:
                print(f"  → [tool] {name}({args})")

            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result_content = f"未知工具：{name}"
            else:
                try:
                    result_content = fn(**args)
                except TypeError as e:
                    result_content = f"参数错误：{e}"
                except Exception as e:
                    result_content = f"工具执行失败：{e}"

            # 如果结果是 dict，转为 JSON 字符串
            if isinstance(result_content, dict):
                result_content = json.dumps(result_content, ensure_ascii=False)

            preview = (result_content or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result_content or '') > 120 else ''}\n")

            # ====== msgs = append(result) ======
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_content,
            })

        # ====== resp = model.chat(msgs, tools) ======
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    # ====== 最终回答 ======
    elapsed = time.time() - t0

    # 如果达到最大迭代且还有 tool_call，强制结束
    if iteration >= max_iterations and msg.tool_calls:
        if verbose:
            print(f"  ⚠️ 达到最大迭代次数 {max_iterations}，强制结束")
        # 让模型直接回答，不再调用工具
        resp = client.chat.completions.create(
            model=model,
            messages=messages + [msg],
            tools=TOOLS_SCHEMA,
            tool_choice="none",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""

    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，{iteration}轮）")

    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "iterations": iteration,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "北京的经纬度是多少？",
    "南京的经纬度是多少？",
    "北京今天的天气怎么样？",
    "南京明天的天气怎么样？",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="链式调用 Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Chain] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()