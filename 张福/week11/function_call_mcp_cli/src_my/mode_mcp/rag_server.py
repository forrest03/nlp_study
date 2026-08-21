"""
rag_server.py — A股年报 RAG 检索 MCP Server（方式二：MCP）

教学重点：
  1. MCP Server 把"现成业务逻辑"封装成协议工具：函数体直接复用 src/rag_backend
     ——零逻辑重复，只加一层 @mcp.tool() 协议装饰
  2. Python 函数签名（类型注解 + docstring）自动生成 JSON Schema 供 LLM 决策
  3. 所有 print/log 必须写 stderr：stdout 是 MCP JSON-RPC 通道，混入普通文本会破坏连接

使用方式（由 run_mcp.py 作为子进程启动，stdio 通信）：
  python mode_mcp/servers/rag_server.py

依赖：
  pip install mcp faiss-cpu numpy openai
  环境变量：DASHSCOPE_API_KEY（Embedding）
"""

import sys
from pathlib import Path

# 让本脚本能 import 项目根的 src/（子进程 cwd 不一定是项目根）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# 注意：用 as 别名导入后端函数，避免下方同名 tool 函数遮蔽后递归调用自己
from src.rag_backend import (  # noqa: E402
    search_annual_report as _search_annual_report,
    list_companies as _list_companies,
)


def log(msg: str):
    # stdout 是协议通道，所有日志必须写 stderr
    print(msg, file=sys.stderr, flush=True)


def _search_tool(
    query: str,
    stock_code: str | None = None,
    year: str | None = None,
    top_k: int = 5,
) -> str:
    """
    在A股年报语料库中检索与问题最相关的段落。

    知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/
    宁德时代(300750)/海康威视(002415)/中国平安(601318)，
    年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。

    Args:
        query:   检索问题。重要：不要包含公司名和年份（已由 stock_code/year 过滤），
                 只用简短财务术语，例如 '营收和净利润'、'研发投入'、'主营业务'。
                 把公司名写进 query 会稀释检索精度。
        stock_code: 可选，按公司过滤，如 '300750'。
        year:       可选，按年份过滤：'2021' / '2022' / '2023'。
        top_k:      返回段落数，默认5，建议不超过10。

    Returns:
        按相关度排序的段落列表，每段含来源（公司、年份、章节、页码）。
    """
    return _search_annual_report(query, stock_code, year, top_k)


def _list_tool() -> str:
    """
    列出年报知识库中收录的所有公司、股票代码与可查年份。
    用于确认目标公司在库内，并获取正确的 stock_code。

    Returns:
        公司列表，含名称、股票代码、可查年份。
    """
    return _list_companies()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument("--transport", "-t", default="stdio", choices=["stdio", "streamable-http"],
                        help="通信方式：stdio 或 streamable-http")
    parser.add_argument("--host", default="127.0.0.1", help="TCP/IP 绑定地址")
    parser.add_argument("--port", "-p", type=int, default=8001, help="TCP/IP 绑定端口")
    args = parser.parse_args()

    mcp = FastMCP("rag-server", host=args.host, port=args.port)
    mcp.add_tool(_search_tool, name="search_annual_report")
    mcp.add_tool(_list_tool, name="list_companies")

    if args.transport == "streamable-http":
        log(f"RAG MCP Server 启动中（streamable-http 模式）... host={args.host} port={args.port}")
        mcp.run(transport="streamable-http")
    else:
        log("RAG MCP Server 启动中（stdio 模式）...")
        mcp.run(transport="stdio")
