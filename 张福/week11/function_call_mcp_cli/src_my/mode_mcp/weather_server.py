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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from src_my.weather_backend import get_weather as _get_weather, get_latlon as _get_latlon, get_weather_latlon as _get_weather_latlon  # noqa: E402


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _weather_tool(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    return _get_weather(city)


def _latlon_tool(city: str) -> str:
    """
    查询指定城市的经纬度。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        所选择城市的经纬度。
    """
    msg = _get_latlon(city)
    return "".join(msg)


def _weather_latlon_tool(lat: str, lon: str) -> str:
    """
    根据经纬度查询天气。

    Args:
        lat: 纬度。
        lon: 精度。

    Returns:
        所选择经纬度的天气。
    """
    msg = _get_weather_latlon(lat, lon)
    return "".join(msg)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weather MCP Server")
    parser.add_argument("--transport", "-t", default="stdio", choices=["stdio", "streamable-http"],
                        help="通信方式：stdio 或 streamable-http")
    parser.add_argument("--host", default="127.0.0.1", help="TCP/IP 绑定地址")
    parser.add_argument("--port", "-p", type=int, default=8002, help="TCP/IP 绑定端口")
    args = parser.parse_args()

    mcp = FastMCP("weather-server", host=args.host, port=args.port)
    mcp.add_tool(_weather_tool, name="get_weather")
    mcp.add_tool(_latlon_tool, name="get_latlon")
    mcp.add_tool(_weather_latlon_tool, name="get_weather_latlon")

    if args.transport == "streamable-http":
        log(f"Weather MCP Server 启动中（streamable-http 模式）... host={args.host} port={args.port}")
        mcp.run(transport="streamable-http")
    else:
        log("Weather MCP Server 启动中（stdio 模式）...")
        mcp.run(transport="stdio")
