"""
weather_server.py — 天气查询 MCP Server（方式二：MCP）

教学重点：
  1. 把 src/weather_backend 的同步函数包成 MCP 工具，加一行装饰器即可
  2. 与 rag_server 共存于不同子进程，由 Host 统一管理——展示 MCP"多 Server 聚合"
  3. 拆分为细粒度工具，支持大模型分步调用

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/weather_server.py

依赖：
  pip install mcp httpx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 导入拆分后的独立函数
from src.weather_backend import (
    search_city as _search_city,
    get_current_weather as _get_current_weather,
    get_forecast as _get_forecast,
)  # noqa: E402


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def search_city(city: str) -> list:
    """
    搜索城市，返回匹配的城市列表（含经纬度、行政级别等信息）。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        匹配的城市列表，每个元素包含：name(城市名), latitude(纬度), longitude(经度),
        feature_code(行政级别代码), population(人口), country(国家), admin1(省/州)
    """
    return _search_city(city)


@mcp.tool()
def get_current_weather(latitude: float, longitude: float) -> dict:
    """
    根据经纬度获取当前天气。

    Args:
        latitude: 纬度（如 26.64）
        longitude: 经度（如 119.31）

    Returns:
        当前天气数据字典，包含：temperature(温度°C), humidity(湿度%), 
        wind_speed(风速km/h), weather_code(天气代码), weather_desc(天气描述)
    """
    return _get_current_weather(latitude, longitude)


@mcp.tool()
def get_forecast(latitude: float, longitude: float, days: int = 3) -> dict:
    """
    根据经纬度获取未来天气预报。

    Args:
        latitude: 纬度（如 26.64）
        longitude: 经度（如 119.31）
        days: 预报天数，默认3天，最多14天

    Returns:
        预报数据字典，包含：latitude, longitude, daily(每日预报列表)
    """
    return _get_forecast(latitude, longitude, days)


# @mcp.tool()
# def get_weather(city: str) -> str:
#     """
#     （聚合方法）查询指定城市的当前天气及未来3天预报。
#     如需更细粒度控制，可分步调用 search_city → get_current_weather → get_forecast。

#     Args:
#         city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

#     Returns:
#         包含温度、湿度、风速、天气状况和3天预报的文字描述。
#     """
#     return _get_weather(city)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")