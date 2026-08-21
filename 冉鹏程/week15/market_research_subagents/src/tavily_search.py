"""Tavily 联网搜索封装（零额外依赖，用标准库 urllib）。

企业信息调查需要实时、可追溯的公开证据。Tavily 返回摘要和来源，本模块会保留来源 URL，
以便主 Agent 在报告中引用和标明信息缺口。
"""

import json
import logging
import os
import urllib.request

from config import load_project_environment

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

load_project_environment()


def tavily_search(query: str, max_results: int = 5, request_id: str = None) -> dict:
    """调用 Tavily 搜索并返回可引用的公开证据。

    参数：query 为长度不超过 300 的检索词；max_results 为结果数；request_id 用于日志追踪。
    返回：包含 answer、results、response_time 的字典；参数非法或调用失败时返回 error 字段。
    """
    normalized_query = _validate_query(query)
    if normalized_query is None:
        return {"error": "查询词必须是 1 至 300 个字符的文本"}
    if not isinstance(max_results, int) or not 1 <= max_results <= 10:
        return {"error": "max_results 必须是 1 至 10 的整数"}
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "未设置 TAVILY_API_KEY"}
    payload = {
        "api_key": api_key,
        "query": normalized_query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        logger.info("tavily_search_started request_id=%s query_length=%s", request_id, len(normalized_query))
        request = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        results = _extract_results(response_data)
        logger.info("tavily_search_completed request_id=%s result_count=%s", request_id, len(results))
        return {"answer": response_data.get("answer") or "", "results": results,
                "response_time": response_data.get("response_time")}
    except Exception as error:
        logger.warning("tavily_search_failed request_id=%s error_type=%s", request_id, type(error).__name__)
        return {"error": f"{type(error).__name__}: {str(error)[:100]}"}


def format_search_result(result: dict) -> str:
    """将 Tavily 结果格式化为带 URL 的证据文本，供 Agent 引用。"""
    if "error" in result:
        return f"搜索失败: {result['error']}"
    parts = []
    if result.get("answer"):
        parts.append(f"摘要: {result['answer']}")
    for index, source in enumerate(result.get("results", []), 1):
        parts.append(f"【S{index}】{source['title']}\nURL: {source['url']}\n"
                     f"摘要: {source['content'][:300]}")
    return "\n".join(parts) if parts else "无结果"


def _validate_query(query: str) -> str | None:
    """校验外部检索词，防止空值或过长输入透传至远程服务。"""
    if not isinstance(query, str):
        return None
    normalized = " ".join(query.split())
    return normalized if 1 <= len(normalized) <= 300 else None


def _extract_results(response_data: dict) -> list[dict]:
    """从 Tavily 响应抽取限定长度的公开来源字段。"""
    return [{"title": item.get("title", ""), "url": item.get("url", ""),
             "content": (item.get("content") or "")[:600]}
            for item in response_data.get("results", [])]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    search_result = tavily_search("小米集团营收 年报")
    print(format_search_result(search_result)[:400])
