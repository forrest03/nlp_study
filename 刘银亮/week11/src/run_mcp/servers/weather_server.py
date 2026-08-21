import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from tools.weather_query import get_weather_data as _get_weather_data, get_geo as _get_geo


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather_data(lat: float, lon: float) -> dict | None:
    """
    根据经纬度获取天气数据。

    Args:
        lat: 纬度
        lon: 经度

    Returns:
        包含 current 和 daily 的天气数据字典；获取失败返回 None
    """
    return _get_weather_data(lat, lon)


@mcp.tool()
def get_geo(city: str) -> dict | None:
    """
    查询指定城市的经纬度信息。

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        包含纬度、经度、城市名称、国家、行政区域的字典；获取失败返回 None
    """
    return _get_geo(city)


if __name__ == "__main__":
    log("Weather MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
    #mcp.run()