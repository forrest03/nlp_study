"""
run_function_call.py - 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     --这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 多轮循环：模型输出 tool_call -> 宿主执行工具 -> 结果以 role=tool 回填 -> 再请求 ...
     只要模型还在调工具就继续，直到给出不带 tool_call 的最终答案（天气条件题天然跨多轮）
  3. 并行工具调用：模型一次输出多个 tool_call（如同时查两个城市天气），宿主逐个执行后一并回填
  4. 工具名 -> 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "宁德现在天气如何？"

  # 内置示例问题（演示天气多轮循环）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai httpx
  环境变量：DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的多轮循环代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 几乎一样，
  差异只在"工具从哪来"和"调用怎么执行"--这正是三者对比的教学点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.weather_backend import geocode, get_weather_by_coords  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 即 deepseek-v4-flash
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


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"--写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "城市名 -> 经纬度。返回 JSON 字符串 {lat, lon, name, country, admin1}；未找到返回 'null'。",
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
            "name": "get_weather_by_coords",
            "description": "经纬度 -> 当前天气及未来3天预报。lat/lon 必填；name/country/admin1 选填（用于报告标题，可从 geocode 结果转发）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "纬度（来自 geocode）"},
                    "lon": {"type": "number", "description": "经度（来自 geocode）"},
                    "name": {"type": "string", "description": "城市名（可选，用于报告标题）"},
                    "country": {"type": "string", "description": "国家（可选）"},
                    "admin1": {"type": "string", "description": "省/州（可选）"},
                },
                "required": ["lat", "lon"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 -> 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"--把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "geocode": geocode,
    "get_weather_by_coords": get_weather_by_coords,
}


# ── 多轮循环 ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气查询助手。查天气分两步：先调 geocode(city) 拿到城市的经纬度"
    "（返回 JSON：lat/lon/name/country/admin1），再调 get_weather_by_coords(lat, lon, name, country, admin1) 取天气预报。"
    "本回合你可以一次调用多个工具（如同时 geocode 两个城市）。"
    "天气查询支持多轮：若问题需要基于一次天气结果做条件判断（如'若气温低于20度则查A城，否则查B城'），"
    "请先查条件所需的城市，看到结果后再决定下一步查哪个城市，不要预先把所有候选城市一次查完。"
)


MAX_ROUNDS = 8  # 安全上限，防止模型无限调用工具


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮循环：提问 -> 模型输出 tool_call -> 执行 -> 回填 -> 再请求 ...
    只要模型还在调用工具就继续，直到给出不带 tool_call 的最终答案；MAX_ROUNDS 触顶强制收尾。
    返回 {answer, tool_calls, elapsed, rounds} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    msg = None
    rounds = 0

    for round_i in range(MAX_ROUNDS):
        rounds = round_i + 1
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型未调用工具 -> 已是最终答案，结束
        if not msg.tool_calls:
            break

        # 【教学时刻 3】：模型输出了 tool_calls -> 逐个执行后端函数
        # 把 assistant 这条带 tool_calls 的消息原样回填，保持上下文
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  -> [tool] {name}({args})  [round {rounds}]")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    # 工具执行！！
                    result = fn(**args)
                    # 后端函数可能返回结构化数据（如 geocode 返回 dict），统一序列化成字符串喂给 LLM
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            # 以 role=tool 把每个工具的结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # 本轮有工具调用：进入下一轮，模型可基于结果继续（再调工具 / 给出最终答案）
    else:
        # 触达轮次上限：强制模型不再调工具，给出最终答案
        if verbose:
            print(f"  ⚠ 达到最大轮次 {MAX_ROUNDS}，强制收尾")
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
        print(f"  -> [llm] 最终回答（{elapsed:.1f}s，共 {rounds} 轮）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed, "rounds": rounds}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    # 天气多轮循环演示：先查宁德，再依气温条件查第二个城市 -> 必然跨 2 轮
    "查一下宁德现在的天气，如果宁德当前气温低于 20 度，就再查哈尔滨的天气，否则再查广州的天气。",
    "北京和上海现在哪个气温更高？",
    "查一下哈尔滨现在的天气。",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Function Call] provider={args.provider} model={model}\n")

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
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
