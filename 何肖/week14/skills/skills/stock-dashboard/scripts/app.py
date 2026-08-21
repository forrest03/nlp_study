# -*- coding: utf-8 -*-
"""
FastAPI 前端服务（新架构）

工作流：
    读取 html_data/ 下由 fetch_stock.py 预生成的 HTML 页面进行静态服务，
    同时提供 json_data/ 下的原始 JSON 查询与列表索引。

路由：
    GET /                                 已生成页面列表
    GET /stock/{company}/{date}           直接返回预生成 HTML
    GET /api/data/{company}/{date}        返回原始 JSON
    GET /static/...                       共享静态资源
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
HTML_DATA_DIR = ROOT_DIR / "html_data"
JSON_DATA_DIR = ROOT_DIR / "json_data"
STATIC_DIR = ROOT_DIR / "static"

HTML_DATA_DIR.mkdir(parents=True, exist_ok=True)
JSON_DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="股票信息看板", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def safe_filename(text: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", unicodedata.normalize("NFKC", str(text))).strip("_")


def find_html_file(company: str, date_str: str) -> Path | None:
    candidate = HTML_DATA_DIR / f"{safe_filename(company)}_{date_str}.html"
    if candidate.exists():
        return candidate
    # 退化：按日期 + company 字段匹配
    target = safe_filename(company)
    for p in HTML_DATA_DIR.glob(f"*_{date_str}.html"):
        try:
            # HTML 首行附近有 __PAGE_TITLE__ 替换后的 title；读取 JSON 更可靠
            json_candidate = JSON_DATA_DIR / f"{p.stem}.json"
            if json_candidate.exists():
                with open(json_candidate, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if safe_filename(payload.get("company", "")) == target:
                    return p
        except (json.JSONDecodeError, OSError):
            continue
    return None


def find_json_file(company: str, date_str: str) -> Path | None:
    candidate = JSON_DATA_DIR / f"{safe_filename(company)}_{date_str}.json"
    if candidate.exists():
        return candidate
    target = safe_filename(company)
    for p in JSON_DATA_DIR.glob(f"*_{date_str}.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if safe_filename(payload.get("company", "")) == target:
                return p
        except (json.JSONDecodeError, OSError):
            continue
    return None


def list_pages() -> list[dict]:
    """扫描 html_data/ 下的所有页面，返回简要清单。"""
    items = []
    for p in sorted(HTML_DATA_DIR.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True):
        # 读取同名 JSON 获取元信息
        json_path = JSON_DATA_DIR / f"{p.stem}.json"
        meta = {"company": p.stem, "stock_code": "", "date": "", "verdict": ""}
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                meta = {
                    "company": payload.get("company", ""),
                    "stock_code": payload.get("stock_code", ""),
                    "date": payload.get("date", ""),
                    "verdict": payload.get("buy_sell", {}).get("verdict", ""),
                }
            except (json.JSONDecodeError, OSError):
                pass
        items.append({**meta, "filename": p.name})
    return items


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    pages = list_pages()
    rows = ""
    if not pages:
        rows = """
        <tr><td colspan='5' class='empty'>尚无生成的页面。请先运行：
            <code>python scripts/fetch_stock.py --company 平安银行 --date 2026-07-28</code>
        </td></tr>"""
    else:
        verdict_color = {
            "看多": "#ff3b3b", "偏多": "#ff7a3b",
            "中性": "#ffd23b",
            "偏空": "#3bff9a", "看空": "#3bff5a",
        }
        for d in pages:
            color = verdict_color.get(d["verdict"], "#7fb3ff")
            link = f"/stock/{quote(d['company'])}/{d['date']}"
            rows += f"""
            <tr>
                <td class='name'><a href='{link}'>{d['company']}</a></td>
                <td>{d['stock_code']}</td>
                <td>{d['date']}</td>
                <td><span class='badge' style='--c:{color}'>{d['verdict'] or '-'}</span></td>
                <td><a class='open' href='{link}'>查看 →</a></td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>股票信息看板 · 页面列表</title>
<style>
    :root{{--bg:#05070d;--panel:#0c111d;--line:#1b2740;--cyan:#00e5ff;--txt:#c9d6e8;--muted:#6b7a96}}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:radial-gradient(1200px 600px at 70% -10%,#0a1430 0%,var(--bg) 60%);
        color:var(--txt);font-family:'Segoe UI','Microsoft YaHei',sans-serif;min-height:100vh;
        display:flex;flex-direction:column;align-items:center;padding:48px 16px}}
    h1{{font-size:28px;letter-spacing:2px;color:var(--cyan);
        text-shadow:0 0 18px rgba(0,229,255,.5);margin-bottom:6px}}
    .sub{{color:var(--muted);margin-bottom:30px;font-size:13px}}
    table{{width:100%;max-width:880px;border-collapse:collapse;
        background:linear-gradient(180deg,var(--panel),#080c16);
        border:1px solid var(--line);border-radius:14px;overflow:hidden;
        box-shadow:0 0 40px rgba(0,229,255,.06)}}
    th,td{{padding:14px 18px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}}
    th{{color:var(--cyan);font-weight:600;letter-spacing:1px;background:rgba(0,229,255,.04)}}
    tr:last-child td{{border-bottom:none}}
    tr:hover td{{background:rgba(0,229,255,.05)}}
    .name a{{color:#eaf2ff;text-decoration:none;font-weight:600}}
    .name a:hover{{color:var(--cyan)}}
    .open{{color:var(--cyan);text-decoration:none}}
    .empty{{color:var(--muted);padding:30px;text-align:center}}
    .empty code{{color:var(--cyan)}}
    .badge{{display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;
        border:1px solid color-mix(in srgb,var(--c) 60%,transparent);
        color:var(--c);background:color-mix(in srgb,var(--c) 12%,transparent)}}
    .hint{{margin-top:24px;color:var(--muted);font-size:12px;max-width:880px;line-height:1.7}}
    .hint code{{color:var(--cyan)}}
</style></head>
<body>
    <h1>◆ 股票信息看板</h1>
    <div class='sub'>DARK · TECH · STOCK DASHBOARD</div>
    <table>
        <thead><tr><th>公司</th><th>股票代码</th><th>日期</th><th>判定</th><th>操作</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    <div class='hint'>
        使用方法：<br>
        1. 获取数据并生成页面：<code>python scripts/fetch_stock.py --company 平安银行 --date 2026-07-28</code><br>
        2. 启动服务：<code>cd scripts && uvicorn app:app --reload</code>，默认端口 8000<br>
        3. 点击「查看」或在浏览器访问 <code>/stock/{{公司}}/{{日期}}</code>
    </div>
</body></html>"""
    return HTMLResponse(html)


@app.get("/stock/{company}/{date}", response_class=HTMLResponse)
def stock_page(company: str, date: str):
    path = find_html_file(company, date)
    if path is None:
        # 提示可能的 JSON 文件
        json_path = find_json_file(company, date)
        if json_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"未找到 {company} 在 {date} 的数据页面。"
                       f"请先运行：python scripts/fetch_stock.py --company {company} --date {date}"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"找到 JSON 数据 ({json_path.name}) 但尚未生成 HTML 页面。"
                       f"请运行：python scripts/fetch_stock.py --company {company} --date {date}"
            )
    return FileResponse(str(path))


@app.get("/api/data/{company}/{date}", response_class=JSONResponse)
def stock_api(company: str, date: str):
    path = find_json_file(company, date)
    if path is None:
        raise HTTPException(status_code=404, detail=f"未找到 {company} 在 {date} 的 JSON 数据。")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
