import sys
import os
import json
import time
from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.weather_query import get_weather_data, get_geo

ALIYUN_API_URL  = os.getenv("ALIYUN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL       = "qwen3.7-plus"

def get_client() -> OpenAI:
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 ALIYUN_API_KEY")
    return OpenAI(api_key=api_key, base_url=ALIYUN_API_URL)

# ────────────────────── 手写工具的 JSON Schema ───────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_geo",
            "description": "根据城市名称获取城市的经纬度信息。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市中文名，如 '宁德'"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_data",
            "description": "根据经纬度获取天气数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "纬度"},
                    "lon": {"type": "number", "description": "经度"},
                },
                "required": ["lat", "lon"],  
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_geo": get_geo,
    "get_weather_data": get_weather_data,
}

SYSTEM_PROMPT = (
    "你是一名旅游博主, 熟悉旅游相关知识以及地理信息。"
    "涉及天气时可以调用 get_weather_data, get_geo 函数。你可以一次调用多个工具。"
)

def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 继续调用直到无工具调用 → 最终回答。
    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    while True:
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
                print(f"  → [tool] {name}({args})")
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
            preview = (str(result)[:120] if result else "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(str(result) or '') > 120 else ''}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Function Call")
    parser.add_argument("--question", "-q", required=True, help="提问内容")
    
    args = parser.parse_args()
    q = args.question

    client = get_client()

    result = run(client, LLM_MODEL, q)
    result["question"] = q
    
    print("\n最终回答：")
    print(result["answer"])
    print()

