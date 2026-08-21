"""
模拟浏览器 Web 搜索（无 Tavily）

用标准库 urllib 伪装浏览器访问 DuckDuckGo HTML 结果页，解析标题/链接/摘要；
可选打开前几条结果页抽取正文与 og:image，供蛋糕商品图文采集。

失败时降级 demo_catalog，保证教学链路可跑通。
"""
from __future__ import annotations

import html as html_lib
import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# 伪装常见桌面 Chrome，降低被搜索引擎拦截概率
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
DDG_HTML = "https://html.duckduckgo.com/html/"


def _http_get(url: str, timeout: int = 20, data: Optional[bytes] = None) -> str:
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://duckduckgo.com/",
        },
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = html_lib.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_ddg_html(page: str, max_results: int) -> list[dict]:
    """解析 DuckDuckGo HTML 结果块。"""
    results = []
    # 结果块：result__a + result__snippet
    blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
        page,
        re.S | re.I,
    )
    for href, title_html, snip_html in blocks:
        title = _strip_tags(title_html)
        snippet = _strip_tags(snip_html)
        url = html_lib.unescape(href)
        # DDG 有时包一层 //duckduckgo.com/l/?uddg=
        m = re.search(r"[?&]uddg=([^&]+)", url)
        if m:
            url = urllib.parse.unquote(m.group(1))
        if not url.startswith("http"):
            continue
        if not title:
            continue
        results.append({"title": title, "url": url, "content": snippet[:600]})
        if len(results) >= max_results:
            break

    if results:
        return results

    # 宽松兜底：只抓 result__a
    for href, title_html in re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S | re.I
    ):
        title = _strip_tags(title_html)
        url = html_lib.unescape(href)
        m = re.search(r"[?&]uddg=([^&]+)", url)
        if m:
            url = urllib.parse.unquote(m.group(1))
        if url.startswith("http") and title:
            results.append({"title": title, "url": url, "content": ""})
        if len(results) >= max_results:
            break
    return results


def _extract_page_meta(url: str) -> dict:
    """模拟浏览器打开结果页，抽取标题/描述/og:image。"""
    try:
        page = _http_get(url, timeout=12)
    except Exception as e:
        return {"error": str(e)[:80]}

    def _meta(prop: str) -> str:
        m = re.search(
            rf'<meta[^>]+(?:property|name)=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
            page,
            re.I,
        )
        if m:
            return html_lib.unescape(m.group(1)).strip()
        m = re.search(
            rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']{re.escape(prop)}["\']',
            page,
            re.I,
        )
        return html_lib.unescape(m.group(1)).strip() if m else ""

    title_m = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
    title = _strip_tags(title_m.group(1)) if title_m else ""
    desc = _meta("og:description") or _meta("description")
    image = _meta("og:image") or _meta("twitter:image")
    # 再抓几张 img src（商品图兜底）
    imgs = []
    if image and image.startswith("http"):
        imgs.append(image)
    for src in re.findall(r'<img[^>]+src=["\'](https?://[^"\']+)["\']', page, re.I):
        if any(x in src.lower() for x in ("logo", "icon", "sprite", "1x1", "pixel")):
            continue
        if src not in imgs:
            imgs.append(src)
        if len(imgs) >= 3:
            break
    text_bits = []
    if desc:
        text_bits.append(desc)
    # 粗抽段落
    for p in re.findall(r"<p[^>]*>(.*?)</p>", page, re.S | re.I)[:5]:
        t = _strip_tags(p)
        if len(t) > 40:
            text_bits.append(t[:200])
    return {
        "title": title,
        "content": " ".join(text_bits)[:600],
        "images": imgs[:3],
    }


def browser_web_search(
    query: str,
    max_results: int = 5,
    fetch_pages: int = 2,
    include_images: bool = True,
) -> dict:
    """模拟浏览器搜索。返回 {answer, results, images, source, response_time}。"""
    t0 = time.time()
    try:
        body = urllib.parse.urlencode({"q": query, "kl": "cn-zh"}).encode("utf-8")
        page = _http_get(DDG_HTML, data=body)
        results = _parse_ddg_html(page, max_results=max_results)
        if not results:
            raise RuntimeError("DuckDuckGo HTML 未解析到结果")

        images: list[str] = []
        if include_images and fetch_pages > 0:
            for item in results[:fetch_pages]:
                meta = _extract_page_meta(item["url"])
                if meta.get("content") and len(meta["content"]) > len(item.get("content") or ""):
                    item["content"] = meta["content"]
                for u in meta.get("images") or []:
                    if u.startswith("http") and u not in images:
                        images.append(u)

        elapsed = round(time.time() - t0, 2)
        return {
            "answer": f"模拟浏览器搜索「{query}」共 {len(results)} 条",
            "results": results,
            "images": images[:8],
            "response_time": elapsed,
            "source": "browser_ddg",
        }
    except Exception as e:
        logger.warning(f"浏览器搜索失败 '{query}': {e}，降级 demo_catalog")
        from demo_catalog import search_demo_catalog

        r = search_demo_catalog(query, max_results=max_results)
        r["error_note"] = f"{type(e).__name__}: {str(e)[:100]}"
        r["response_time"] = round(time.time() - t0, 2)
        return r


# 兼容旧导入名
def web_search(query: str, max_results: int = 5, include_images: bool = True) -> dict:
    return browser_web_search(
        query, max_results=max_results, include_images=include_images
    )


def format_search_result(r: dict) -> str:
    """把搜索返回格式化成喂给 LLM 的文本（含图片链接）。"""
    if "error" in r and not r.get("results"):
        return f"搜索失败: {r['error']}"
    parts = []
    src = r.get("source") or ""
    if src == "demo_catalog":
        parts.append("数据来源: demo_catalog（演示降级）")
    elif src == "browser_ddg":
        parts.append("数据来源: 模拟浏览器 Web 搜索（DuckDuckGo HTML）")
    if r.get("error_note"):
        parts.append(f"联网备注: {r['error_note']}")
    if r.get("answer"):
        parts.append(f"摘要: {r['answer']}")
    for i, res in enumerate(r.get("results", []), 1):
        parts.append(
            f"[{i}] {res['title']}\n    URL: {res['url']}\n    {(res.get('content') or '')[:300]}"
        )
    imgs = r.get("images") or []
    if imgs:
        parts.append("图片链接:")
        for j, u in enumerate(imgs, 1):
            parts.append(f"  [图{j}] {u}")
    return "\n".join(parts) if parts else "无结果"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = browser_web_search("生日蛋糕 商品 价格 口味", max_results=4, fetch_pages=1)
    print(format_search_result(r)[:1200])
