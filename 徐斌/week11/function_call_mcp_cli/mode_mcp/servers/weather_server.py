"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. 天气拆成 geocode_city + get_weather_by_coords 两步，由 Host 多轮循环串联
  3. 由 Host 经 stdio 连接本 Server，完成工具发现与 call_tool

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from src.weather_backend import (  # noqa: E402
    geocode_city as _geocode_city,
    get_weather_by_coords as _get_weather_by_coords,
)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def geocode_city(city: str) -> str:
    """
    将城市/地区名解析为经纬度坐标。

    查天气时必须先调用本工具拿到 latitude/longitude，
    再调用 get_weather_by_coords；不要跳过本步。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        JSON 字符串，含 latitude / longitude / name / country / admin1。
    """
    return _geocode_city(city)


@mcp.tool()
def get_weather_by_coords(
    latitude: float,
    longitude: float,
    location_name: str = "",
) -> str:
    """
    按经纬度查询当前天气及未来3天预报。

    Args:
        latitude: 纬度，必须来自 geocode_city 的返回结果。
        longitude: 经度，必须来自 geocode_city 的返回结果。
        location_name: 可选，展示用地点名，如 geocode 返回的 name。

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather_by_coords(latitude, longitude, location_name)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
