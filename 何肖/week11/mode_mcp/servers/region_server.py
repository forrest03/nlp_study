import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 用 as 别名避免同名 tool 函数遮蔽后端函数导致递归
from src.region_backend import get_region as _get_region  # noqa: E402


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


mcp = FastMCP("region-server")


@mcp.tool()
def get_region(city: str) -> str:
    """
    查询指定城市的经纬度

    Args:
        city: 城市中文名，如 '宁德'、'北京'。同名地名会自动取行政级别更高的（如福建宁德而非西藏宁德）。

    Returns:
        城市的经纬度
    """
    return _get_region(city)


if __name__ == "__main__":
    log("Region MCP Server 启动中（stdio 模式）...")
    mcp.run(transport="stdio")
