"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. 由 run_mcp.py 作为子进程启动，stdio 通信——展示 MCP 协议接入

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归（tool 函数体内调别名）
from src.weather_backend import (  # noqa: E402
    geocode as _geocode,
    get_weather_by_coords as _get_weather_by_coords,
)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def geocode(city: str) -> str:
    """
    城市名 -> 经纬度。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。

    Returns:
        JSON 字符串 {"lat", "lon", "name", "country", "admin1"}；未找到返回 "null"。
    """
    loc = _geocode(city)
    return json.dumps(loc, ensure_ascii=False) if loc is not None else "null"


@mcp.tool()
def get_weather_by_coords(lat: float, lon: float, name: str = "", country: str = "", admin1: str = "") -> str:
    """
    经纬度 -> 当前天气及未来3天预报。lat/lon 必填；name/country/admin1 选填（用于报告标题）。

    Args:
        lat: 纬度
        lon: 经度
        name: 城市名（可选，用于报告标题）
        country: 国家（可选）
        admin1: 省/州（可选）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather_by_coords(lat, lon, name, country, admin1)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
