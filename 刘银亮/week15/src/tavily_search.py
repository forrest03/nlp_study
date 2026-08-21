"""
Tavily 联网搜索工具（通过 requests 直接调用 Tavily API）
"""

import os
import requests

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, max_results: int = 5) -> dict:
    """调用 Tavily Search API，返回原始 JSON"""
    if not TAVILY_API_KEY:
        return {"error": "TAVILY_API_KEY 未设置"}
    try:
        resp = requests.post(
            TAVILY_URL,
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": max_results},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def format_search_result(result: dict) -> str:
    """将 Tavily 返回结果格式化为 LLM 可读文本"""
    if "error" in result:
        return f"搜索失败: {result['error']}"
    if not result.get("results"):
        # 检查是否有 answer 字段（Tavily 有时直接返回答案）
        answer = result.get("answer", "")
        return f"搜索结果: {answer}" if answer else "未找到相关结果"
    lines = []
    for r in result.get("results", [])[:5]:
        title = r.get("title", "")
        url = r.get("url", "")
        content = (r.get("content") or "")[:300]
        lines.append(f"- {title}\n  URL: {url}\n  {content}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    query = "2024年比亚迪汉2024款性能参数"
    result = tavily_search(query)
    print(format_search_result(result))