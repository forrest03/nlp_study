import os
import sys
import json
import time
import argparse

from openai import OpenAI

# ==================== 模块1：实现独立天气工具函数 ====================
from tools import geocode, get_weather


# ==================== 模块2：定义工具元数据（tools 参数）+ 派发表 ====================
# 功能：按照模型的 function call 规范，定义两个工具的对外描述，并建立"工具名→函数"派发表
#
# 【重要】tools.py 里实际有两个工具，必须都暴露给模型，否则作业退化成单工具模式：
#   - geocode(city_name: str) → dict  ：城市名 → 经纬度
#   - get_weather(latitude: float, longitude: float) → dict ：经纬度 → 天气
#
# 工具1：geocode
#   - name：必须与函数名完全一致 "geocode"
#   - description：说明"城市名→经纬度"，并提示模型"用户问经纬度时直接用本工具；
#     用户问天气时先用本工具拿经纬度，再调 get_weather（链式）"
#   - parameters：
#       city_name（string，必填）：城市中文名，如 "宁德"、"北京"
#   - required：["city_name"]
#
# 工具2：get_weather
#   - name："get_weather"
#   - description：说明"经纬度→天气"，并提示"若用户只给城市名，请先调 geocode 拿经纬度再调本工具"
#   - parameters：
#       latitude（number，必填）：纬度，如 39.9
#       longitude（number，必填）：经度，如 116.4
#   - required：["latitude", "longitude"]
#
# 【铁律】schema 的参数名/类型必须与 tools.py 函数签名完全一致，否则模型按 schema
#   生成的参数传给函数时会 TypeError。注意：tools.py 里 get_weather 没有 city、没有 date。
#
# 派发表 TOOL_DISPATCH（dict）：
#   - key 为工具名字符串，value 为对应函数对象（不是字符串）
#   - 例：{"geocode": geocode, "get_weather": get_weather}
#   - 模块5.3 执行工具时通过 TOOL_DISPATCH[name] 拿到函数调用

# 工具元数据（tools 参数）：定义两个工具的对外描述，供模型识别与调用
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "城市名→经纬度。当用户询问城市的经纬度时直接使用本工具；当用户询问天气时，先使用本工具获取该城市的经纬度，再调用 get_weather 工具查询天气（链式调用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "城市中文名，如 '宁德'、'北京'"
                    }
                },
                "required": ["city_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "经纬度→天气。根据经纬度查询天气信息。若用户只提供城市名，请先调用 geocode 工具获取经纬度，再调用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "纬度，如 39.9"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "经度，如 116.4"
                    }
                },
                "required": ["latitude", "longitude"]
            }
        }
    }
]

# 派发表：工具名→函数对象，用于模块5.3执行工具时查找对应函数
TOOL_DISPATCH = {
    "geocode": geocode,
    "get_weather": get_weather
}


# ==================== 模块3：初始化模型与API客户端 ====================
# 功能：配置支持 function call 的模型及客户端
# 要求：
#   - 用 OpenAI SDK（pip install openai），通过 OpenAI 兼容协议接入 DeepSeek
#   - 定义 PROVIDERS 字典，含两个 provider：
#       "deepseek"：api_key 从环境变量 DEEPSEEK_API_KEY 读取，
#                   base_url="https://api.deepseek.com"，model="deepseek-chat"
#       "dashscope"：api_key 从环境变量 DASHSCOPE_API_KEY 读取，
#                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"，model="qwen-plus"
#   - build_client(provider) 函数：从 PROVIDERS 取配置，若 api_key 为空则报错退出；
#     返回 (OpenAI(api_key, base_url), model) 元组
#   - 参考 mode_function_call/run_function_call.py 第 48-67 行，可直接复用其写法

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-max",
    },
}


def build_client(provider="deepseek"):
    """根据 provider 名称构建 OpenAI 客户端与模型名"""
    cfg = PROVIDERS.get(provider)
    if cfg is None:
        print(f"未知 provider: {provider}")
        sys.exit(1)
    api_key = cfg.get("api_key", "")
    if not api_key:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    model = cfg["model"]
    return client, model

# ==================== 模块4：构建初始对话请求消息 ====================
# 功能：构建多轮循环的起始消息列表
# 要求：
#   - messages 为 list，含两条初始消息：
#       ① role="system"：SYSTEM_PROMPT，告诉模型"你是天气助手，有两个工具 geocode 和
#          get_weather，必要时可链式调用（先 geocode 拿经纬度，再 get_weather 查天气），
#          只依据工具返回数据作答，不要编造"
#       ② role="user"：用户的原始查询字符串
#   - system 消息是让模型知道"可以链式调用"的关键，不能省略
#   - 参考 run_agent.py 第 119-124 行的 SYSTEM_PROMPT 写法
# 定义系统提示词：告知模型可用工具、链式调用方式、以及只依据工具数据作答的要求
SYSTEM_PROMPT = "你是一个天气助手，拥有 geocode 和 get_weather 两个工具。必要时可以链式调用：先使用 geocode 获取城市的经纬度，再使用 get_weather 查询天气。请严格依据工具返回的数据进行回答，不要编造信息。"


# ==================== 模块5：实现多轮工具调用循环核心逻辑 ====================
# 功能：驱动模型与工具的多轮交互，直到获得最终的可读回复
#
# 【防御】模块顶部定义 MAX_STEPS = 10，防止模型无限循环调用工具
#
# 循环结构：for step in range(1, MAX_STEPS + 1):
#
#   5.1 调用模型：client.chat.completions.create(
#         model=model, messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto")
#       取 resp.choices[0].message
#
#   5.2 解析响应：判断 msg.tool_calls 是否为空
#       - 若为空 → 模型已给出最终文本回复，msg.content 即最终答案，break 退出循环
#       - 若非空 → 模型本轮要调工具，进入 5.3
#       - 同时把这条带 tool_calls 的 assistant 消息原样 append 到 messages（保持上下文）
#
#   5.3 执行工具：遍历 msg.tool_calls，对每个 tc：
#       - name = tc.function.name
#       - args = json.loads(tc.function.arguments or "{}")   ← 模型给的参数是 JSON 字符串
#       - fn = TOOL_DISPATCH.get(name)
#       - 若 fn 为 None → result = "未知工具：{name}"
#       - 否则 try: result = fn(**args)
#                except TypeError: result = "参数错误：{e}"
#                except Exception: result = "工具执行失败：{e}"
#       - 记录到 tool_call_log（用于统计与调试）
#
#   5.4 追加历史：对每个工具结果，append 一条 role="tool" 消息：
#       - tool_call_id：必须等于 tc.id（OpenAI 协议要求一一对应）
#       - content：【重要】tools.py 返回 dict，但 content 要求是字符串，
#         必须用 json.dumps(result, ensure_ascii=False) 转字符串再回填
#       - 注意：本步在 for tc 循环内逐个回填，不是循环外批量
#
#   5.5 循环继续：回到 for 顶部，模型带上完整历史（含工具结果）再次决策，
#       直到模型不再调工具（输出 content）或达到 MAX_STEPS
#
# 循环结束后返回 {"answer": 最终文本, "tool_calls": tool_call_log, "steps": step, "elapsed": 耗时}
# 防御：最大循环步数，防止模型无限循环调用工具
MAX_STEPS = 10

def run(client, model, messages, tools_schema, tool_dispatch):
    """多轮工具调用循环核心函数，驱动模型与工具的多轮交互"""
    tool_call_log = []
    start_time = time.time()
    answer = ""

    for step in range(1, MAX_STEPS + 1):
        # 5.1 调用模型
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        msg = resp.choices[0].message

        # 5.2 解析响应：判断模型是否需要调用工具
        if not msg.tool_calls:
            # 模型已给出最终文本回复
            answer = msg.content
            break

        # 模型本轮要调工具，把带 tool_calls 的 assistant 消息追加到 messages
        messages.append({
            "role": "assistant",
            "content": msg.content if msg.content else "",
            "tool_calls": msg.tool_calls
        })

        # 5.3 执行工具：遍历模型返回的每个工具调用
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            fn = tool_dispatch.get(name)

            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"

            tool_call_log.append({
                "name": name,
                "args": args,
                "result": result
            })

            # 5.4 追加工具结果到 messages（tool 消息，需与 tc.id 一一对应）
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
    if not answer:
        answer = "(达到最大步数，模型未给出最终回答)"

    elapsed = time.time() - start_time
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "steps": step,
        "elapsed": elapsed
    }

# ==================== 模块6：封装对外调用入口函数 ====================
# 功能：将模块3-5的逻辑封装为可复用的调用接口
# 要求：
#   - 函数签名：weather_assistant(user_query: str, provider: str = "deepseek") -> str
#   - 内部依次：build_client(provider) → 构建 messages → 调模块5的 run() → 返回 answer 文本
#   - 返回值是最终的天气回复字符串（不是 dict），方便上层直接 print
#   - 另可写 main()：用 argparse 接 --question/-q、--demo、--provider 参数，
#     --demo 跑三个内置示例问题（链式/单工具geocode/单工具get_weather 三种形态），
#     参考 run_agent.py 第 200-228 行的 DEMO_QUESTIONS 与 main 写法
def weather_assistant(user_query: str, provider: str = "deepseek") -> str:
    """封装对外调用入口：接收用户查询，返回天气回复字符串"""
    if not user_query or not user_query.strip():
        return "请输入有效的查询问题。"
    client, model = build_client(provider)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    result = run(client, model, messages, TOOLS_SCHEMA, TOOL_DISPATCH)
    return result["answer"]


# 内置示例问题：涵盖链式调用、单工具geocode、单工具get_weather 三种形态
DEMO_QUESTIONS = [
    "北京今天天气怎么样？",          # 链式调用：先 geocode → 再 get_weather
    "宁德的经纬度是多少？",          # 单工具：仅 geocode
    "纬度39.9，经度116.4的天气如何？",  # 单工具：仅 get_weather
]


def main():
    """命令行入口：支持 --question/-q 单问题查询、--demo 跑示例、--provider 选择模型"""
    parser = argparse.ArgumentParser(description="天气助手")
    parser.add_argument("--question", "-q", type=str, default="", help="用户输入的天气查询问题")
    parser.add_argument("--demo", action="store_true", help="运行内置示例问题")
    parser.add_argument("--provider", type=str, default="deepseek", help="选择模型提供者（deepseek/dashscope）")
    args = parser.parse_args()

    if args.demo:
        # 跑三个内置示例问题
        for q in DEMO_QUESTIONS:
            print(f"\n{'='*60}")
            print(f"【示例问题】{q}")
            print(f"{'='*60}")
            try:
                answer = weather_assistant(q, provider=args.provider)
                print(f"【回答】{answer}")
            except Exception as e:
                print(f"【错误】{e}")
        return

    if args.question:
        # 单问题查询
        try:
            answer = weather_assistant(args.question, provider=args.provider)
            print(answer)
        except Exception as e:
            print(f"错误：{e}", file=sys.stderr)
        return

    # 未指定 --question 也未指定 --demo，打印帮助提示
    parser.print_help()


if __name__ == "__main__":
    main()
