"""
registry.py — 工具注册表

集中维护两件东西：
  - TOOLS_SCHEMA: OpenAI function calling 格式的工具描述，喂给 LLM
  - TOOL_DISPATCH: 工具名 → Python 函数 的映射，供 Runner 调用

新增工具三步走：
  1. 在 tools/ 下新建 <your_tool>.py
  2. 在本文件 TOOLS_SCHEMA 加 schema
  3. 在 TOOL_DISPATCH 加映射
"""

from tools.weather import get_weather
from tools.time_tool import get_time

# ── 工具 schema（喂给 LLM） ─────────────────────────────────────────────────
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '北京'、'上海'。"
                "若用户问「两个城市哪个更适合户外」之类的比较问题，请对涉及的每个城市各调一次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，如 '北京'、'上海'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": (
                "查询指定时区的当前时间。时区用 IANA 名，如 'Asia/Shanghai'、'UTC'。"
                "当用户问「这周末」「明天」「现在几点」等依赖时间的问题时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {
                        "type": "string",
                        "description": "IANA 时区名，如 'Asia/Shanghai'、'UTC'、'America/New_York'",
                    },
                },
                "required": ["zone"],
            },
        },
    },
]

# ── 工具 dispatch（程序内调用） ─────────────────────────────────────────────
TOOL_DISPATCH = {
    "get_weather": get_weather,
    "get_time":    get_time,
}
