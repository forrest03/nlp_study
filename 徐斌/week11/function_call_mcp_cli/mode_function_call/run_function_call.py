"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 多轮闭环：模型输出 tool_call → 宿主执行 → role=tool 回填 → 再请求模型
     ——直到模型不再调用工具（或达到轮次上限），生成最终回答
  3. 天气拆成两步：geocode_city（地区→坐标）→ get_weather_by_coords（坐标→天气）
     ——演示"依赖前一步结果"的串行循环调用
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "宁德现在天气怎么样？"

  # 内置示例问题（演示天气串行循环）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的 LLM 循环代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 几乎一样，
  差异只在"工具从哪来"和"调用怎么执行"——这正是三者对比的教学点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.weather_backend import geocode_city, get_weather_by_coords  # noqa: E402

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

MAX_TOOL_ROUNDS = 5


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "geocode_city",
            "description": (
                "将城市/地区名解析为经纬度坐标。"
                "查天气时必须先调用本工具拿到 latitude/longitude，"
                "再调用 get_weather_by_coords；不要跳过本步。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": (
                "按经纬度查询当前天气及未来3天预报。"
                "latitude/longitude 必须来自 geocode_city 的返回结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，来自 geocode_city"},
                    "longitude": {"type": "number", "description": "经度，来自 geocode_city"},
                    "location_name": {
                        "type": "string",
                        "description": "可选，展示用地点名，如 geocode 返回的 name 或 '中国 福建 宁德'",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────

TOOL_DISPATCH = {
    "geocode_city": geocode_city,
    "get_weather_by_coords": get_weather_by_coords,
}


# ── 多轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气助手。回答天气相关问题时必须分两步调用工具："
    "先调用 geocode_city 获取坐标，再根据返回的 latitude/longitude 调用 get_weather_by_coords；"
    "不要臆造坐标或天气数据，只依据工具返回结果作答。"
    "若某工具依赖另一步的结果，请先调前者，等结果回填后再调后者。"
)


def _execute_tool(name: str, args: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"未知工具：{name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"参数错误：{e}"
    except Exception as e:
        return f"工具执行失败：{e}"


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮闭环：提问 →（模型 tool_call → 执行 → 回填）× N → 最终回答。
    天气场景典型路径：第1轮 geocode_city → 第2轮 get_weather_by_coords → 最终回答。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
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
                print(f"  → [tool] {name}({args})")
            result = _execute_tool(name, args)
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
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


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德现在天气怎么样？",
    "北京今天天气如何？另外未来三天呢？",
    "对比一下上海和深圳现在的天气。",
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
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
