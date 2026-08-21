
"""
八斗智能助手 - 使用新版 openai SDK，支持工具调用
"""

from openai import OpenAI
import os
import json

# 导入我们自己的工具函数
from query_geo import query_geo
from query_weater import query_weater as query_weather_by_coords

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except ImportError:
    pass

# 从环境变量获取配置
base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")
api_key = os.getenv("API_KEY")

# 如果环境变量没设置，使用临时值（仅用于测试，生产环境请使用环境变量）
if not api_key:
    api_key = "xxxx"

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# 定义工具 schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_geo",
            "description": "根据输入的城市名称查询经纬度坐标",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "城市名称，支持中文，例如 '北京'、'上海'、'广州'",
                    },
                },
                "required": ["city_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "根据经纬度查询当前天气和未来3天预报",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "纬度",
                    },
                    "lon": {
                        "type": "number",
                        "description": "经度",
                    },
                    "location_name": {
                        "type": "string",
                        "description": "可选的地点名称，用于输出显示，例如 '北京'",
                    },
                },
                "required": ["lat", "lon"],
            },
        },
    },
]

def execute_tool(tool_call):
    """
    执行工具调用
    """
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    print(f"正在执行工具：{function_name}，参数：{function_args}")

    try:
        if function_name == "query_geo":
            lat, lon = query_geo(function_args["city_name"])
            return json.dumps({
                "latitude": lat,
                "longitude": lon,
                "city": function_args["city_name"]
            })
        elif function_name == "query_weather":
            location_name = function_args.get("location_name", "")
            result = query_weather_by_coords(
                lat=function_args["lat"],
                lon=function_args["lon"],
                location_name=location_name
            )
            return json.dumps({
                "weather_report": result
            })
        else:
            return json.dumps({"error": f"未知工具：{function_name}"})
    except Exception as e:
        return json.dumps({"error": f"工具执行失败：{e}"})

def fq(user_input):
    """
    发送请求到大模型，支持多轮工具调用
    """
    try:
        messages = [
            {"role": "system", "content": "你是一名全能助手，能够回答用户的问题。如果需要查询天气或地理位置信息，可以调用相关工具。"},
            {"role": "user", "content": user_input},
        ]

        max_iterations = 10  # 防止无限循环
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # 发送请求
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                max_tokens=1024,
                temperature=0.5,
            )

            assistant_message = response.choices[0].message

            # 如果没有工具调用，直接返回结果
            if not assistant_message.tool_calls:
                return assistant_message.content

            # 如果有工具调用，执行工具
            print(f"第 {iteration} 轮工具调用")
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_response = execute_tool(tool_call)
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_response,
                })

        # 超出最大迭代次数
        return "抱歉，工具调用次数过多，无法完成请求。"

    except Exception as e:
        return f"请求出错：{e}"

# 定义一个用户交换入口，询问用户想问什么问题
def user_exchange():
    #构建欢迎信息
    print("#"*80)
    #居中显示
    print("欢迎使用八斗智能助手".center(80))
    print("#"*80)
    user_input = input("请输入您的问题：")
    #构建问答
    print("正在思考中...")
    answer = fq(user_input)
    print("\n回答：")
    print(answer)
    return answer

user_exchange()
