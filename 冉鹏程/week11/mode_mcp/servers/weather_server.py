"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. 与 rag_server 共存于不同子进程，由 Host 统一管理——展示 MCP"多 Server 聚合"

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from src.weather_backend import get_location as _get_location  # noqa: E402
from src.weather_backend import get_weather as _get_weather  # noqa: E402


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def get_location(city: str) -> dict[str, Any]:
    """
    将城市名称解析为经纬度。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地点会优先选择行政等级较高的候选。

    Returns:
        统一响应：{"success": bool, "data": object | null, "errMsg": str}。
        成功时 data 中包含 latitude、longitude 和 timezone。
    """
    return _get_location(city)


@mcp.tool()
def get_weather(latitude: float, longitude: float, timezone: str = "auto") -> dict[str, Any]:
    """按经纬度查询天气，不接受城市名称。

    Args:
        latitude: 纬度，范围 -90 至 90。
        longitude: 经度，范围 -180 至 180。
        timezone: IANA 时区，默认 auto。

    Returns:
        统一响应。成功时 data 包含当前天气和未来三天预报；失败时 data 为 null。
    """
    return _get_weather(latitude, longitude, timezone)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
