"""Tavily 搜索工具。使用 Python 标准库，不依赖 Tavily SDK。"""

import json
import ssl
import urllib.request

try:
    from config import TAVILY_API_KEY
except ImportError:
    TAVILY_API_KEY = ""


TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query, max_results=3):
    """返回摘要和来源；失败也返回字典，不让 Agent 循环崩溃。"""
    if not TAVILY_API_KEY:
        return {"error": "config.py 中没有设置 TAVILY_API_KEY"}

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,
    }
    try:
        request = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            if "CERTIFICATE_VERIFY_FAILED" not in str(error):
                raise
            # 教学网络可能替换证书；只对固定的 Tavily 地址做兼容处理。
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(request, timeout=20, context=context) as response:
                data = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        return {"error": type(error).__name__ + ": " + str(error)[:160]}

    results = []
    for item in data.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": (item.get("content") or "")[:500],
        })
    return {"answer": data.get("answer") or "", "results": results}


def format_search_result(result):
    """把搜索响应转换为 ReAct 的 Observation 文本。"""
    if "error" in result:
        return "搜索失败: " + result["error"]
    lines = []
    if result["answer"]:
        lines.append("摘要: " + result["answer"])
    for index, item in enumerate(result["results"], start=1):
        lines.append("[" + str(index) + "] " + item["title"])
        lines.append(item["content"])
        lines.append("来源: " + item["url"])
    return "\n".join(lines) if lines else "无搜索结果"
