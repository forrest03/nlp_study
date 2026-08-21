"""
run_function_call_loop.py — 方式一（循环版）：Function Call 多轮 ReAct 循环

教学重点：
  1. 单轮闭环的局限：run_function_call.py 只允许"模型调一次工具 → 看结果 → 出答案"，
     一旦模型想"先查年报、再根据结果决定要不要换个查询再查一次"，就做不到——
     ARCHITECTURE.md 第 99 行点名了这是 Agent 多轮循环要解决的问题。
  2. 循环调用的核心：把 create → tool_calls → 执行 → 回填 → 再 create 包成一个 while 循环，
     每轮模型自己决定"继续调工具"还是"给出最终答案"（无 tool_calls 即退出循环）。
  3. 安全护栏：MAX_ITER 防止模型陷入无限调工具的死循环——这是 Agent 循环必备的兜底。
  4. 工具仍走 Function Call：本文件复用 src/ 后端，只把"协议层"从单轮改成多轮，业务逻辑零改动。

与原版 run_function_call.run() 的唯一差异（仅 protocol 层）：
  · run_function_call.run() ：if msg.tool_calls → 执行 → 再 create 一次 → 结束
  · 本文件 run()            ：while msg.tool_calls → 执行 → 再 create → 直到无 tool_calls

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call_loop/run_function_call_loop.py --question "宁德时代总部宁德的天气如何？如果下雨，推荐一个南方城市并查它的天气"

  # 内置示例问题（演示多轮循环）
  python mode_function_call_loop/run_function_call_loop.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_weather  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

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


# ── 工具 schema 与 dispatch（与 run_function_call.py 一致，便于横向对比）───

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/"
                "宁德时代(300750)/海康威视(002415)/中国平安(601318)，"
                "年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 参数过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'、'主营业务'。"
                            "把公司名写进 query 会稀释检索精度。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '300750'。不传则跨公司检索",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤：'2021' / '2022' / '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5，建议不超过10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。用于确认目标公司在库内。",
            "parameters": {"type": "object", "properties": {}},
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
    },
]

TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_weather": get_weather,
}


# ── 多轮循环 ───────────────────────────────────────────────────────────────

# 循环上限：防止模型陷入"一直调工具不出答案"的死循环——这是 Agent 循环的必备护栏
MAX_ITER = 10

SYSTEM_PROMPT = (
    "你是一名金融分析助手，具备多轮工具调用能力。"
    "回答 A 股年报问题时必须先调用 search_annual_report 工具检索年报原文，只依据工具返回的段落作答，不要编造。"
    "知识库仅含 5 家公司：贵州茅台/五粮液/宁德时代/海康威视/中国平安，年份 2021-2023；不在库内的请明确告知。"
    "涉及天气时调用 get_weather。"
    "你可以分多轮调用工具：先查一个城市/公司，根据结果再决定是否继续查询其他城市/公司。"
    "一旦信息足够回答用户问题，请直接给出最终答案，不要再调用工具。"
)


def _execute_tool_call(tc, verbose: bool, iteration: int) -> tuple[str, dict]:
    """执行单个 tool_call，返回 (结果字符串, 日志条目)。"""
    name = tc.function.name
    args = json.loads(tc.function.arguments or "{}")
    log_entry = {"name": name, "args": args, "round": iteration}
    if verbose:
        print(f"  → [round {iteration}] [tool] {name}({args})")

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
    return result, log_entry


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮循环：提问 → 循环{ 模型输出 tool_call → 执行 → 回填 → 再请求 } → 最终回答。
    与 run_function_call.run() 的唯一差别：把"if msg.tool_calls"改成"while msg.tool_calls"。
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
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型本轮没有 tool_calls = 已生成最终答案，退出循环
        if not msg.tool_calls:
            if verbose:
                print(f"  → [round {iteration}] 模型未调用工具，直接作答\n")
            break

        # 有 tool_calls = 执行后回填，进入下一轮让模型决定是否继续
        messages.append(msg)
        for tc in msg.tool_calls:
            result, log_entry = _execute_tool_call(tc, verbose, iteration)
            tool_call_log.append(log_entry)
            # 每个 tool_call 的结果以 role=tool 回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # 循环回到 while 顶部：再次请求模型，它可以继续调工具或给出答案
    else:
        # while 正常走完 MAX_ITER 次仍没退出——强制再请求一次拿最终答案
        if verbose:
            print(f"  → [!] 达到 MAX_ITER={MAX_ITER}，强制收尾\n")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="none",  # 强制模型不再调工具，必须出文本答案
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

# 这些问题设计上需要"多轮"才能答好：单轮闭环保留的旧版会答不全或被迫一次性硬塞
DEMO_QUESTIONS = [
    # 多轮 1：天气条件分支——模型要先查宁德天气，再根据结果决定要不要查南方城市
    "宁德时代总部宁德的天气如何？如果下雨，推荐一个南方城市并查它的天气。",
    # 多轮 2：依次查多个城市——单轮并行能做，但循环版让模型自己决定下一站
    "帮我依次查北京、上海、广州的天气，并告诉我哪座城市最适合明天出门。",
    # 多轮 3：RAG + 天气联动——先查年报拿到总部城市，再查该城市天气
    "宁德时代的总部在哪？查一下总部所在地的天气。",
    # 多轮 4：对比 + 兜底——单轮的 ARCHITECTURE 点名的"想再检索一次细化"场景
    "对比贵州茅台和五粮液2023年的营收，如果数据不够清楚可以再查一次。",
]


def main():
    global MAX_ITER
    import argparse
    parser = argparse.ArgumentParser(description="方式一（循环版）：Function Call 多轮 ReAct")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集（演示多轮循环）")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-iter", type=int, default=MAX_ITER, help=f"循环上限（默认 {MAX_ITER}）")
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    MAX_ITER = args.max_iter

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call Loop] provider={args.provider} model={model} max_iter={MAX_ITER}\n")

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
