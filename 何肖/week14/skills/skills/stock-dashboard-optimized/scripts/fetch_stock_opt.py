# -*- coding: utf-8 -*-
"""
优化版股票信息获取脚本

优化点：
1. 股票代码缓存 - 首次查询后保存到本地JSON，避免重复加载全市场列表
2. 快速重试 - 缩短退避时间，减少等待
3. 精简输出 - 仅返回结构化摘要
4. 模块导入 - 支持被主进程直接import调用，避免subprocess开销
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
JSON_DATA_DIR = ROOT_DIR / "json_data"
HTML_DATA_DIR = ROOT_DIR / "html_data"
CACHE_DIR = ROOT_DIR / ".cache"
TEMPLATE_PATH = SCRIPT_DIR / "dashboard_template.html"

for d in (JSON_DATA_DIR, HTML_DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

STOCK_CODE_CACHE = CACHE_DIR / "stock_codes.json"


def _load_code_cache() -> dict:
    if STOCK_CODE_CACHE.exists():
        try:
            with open(STOCK_CODE_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_code_cache(cache: dict) -> None:
    with open(STOCK_CODE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def _build_code_cache() -> dict:
    print("  [cache] 首次加载股票代码映射表（约5000条）...")
    df = ak.stock_info_a_code_name()
    cache = {}
    for _, row in df.iterrows():
        code = str(row["code"])
        name = str(row["name"])
        cache[name.lower()] = {"code": code, "name": name}
    _save_code_cache(cache)
    print(f"  [cache] 已缓存 {len(cache)} 只股票代码")
    return cache


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", "", text).lower()


def safe_filename(text: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", text).strip("_")


def resolve_stock_code(company: str) -> tuple[str, str]:
    cache = _load_code_cache()
    if not cache:
        cache = _build_code_cache()

    target = _normalize(company)

    # 精确匹配
    if target in cache:
        entry = cache[target]
        return entry["code"], entry["name"]

    # 模糊匹配：包含
    for name_lower, entry in cache.items():
        if target in name_lower:
            return entry["code"], entry["name"]

    # 反向包含
    for name_lower, entry in cache.items():
        if name_lower in target:
            return entry["code"], entry["name"]

    # 重新构建缓存（可能有更新）
    cache = _build_code_cache()
    if target in cache:
        entry = cache[target]
        return entry["code"], entry["name"]

    raise ValueError(f"未找到匹配的股票：{company}")


def market_prefix(code: str) -> str:
    c = code.strip()
    if c.startswith("6"): return "sh"
    if c.startswith(("0", "3")): return "sz"
    if c.startswith(("8", "4")): return "bj"
    return "sz"


def fetch_min_kline_sina(code: str, date_str: str) -> pd.DataFrame:
    symbol = f"{market_prefix(code)}{code}"
    df = ak.stock_zh_a_minute(symbol=symbol, period="30", adjust="qfq")
    df = df.rename(columns={
        "day": "时间", "open": "开盘", "high": "最高",
        "low": "最低", "close": "收盘", "volume": "成交量", "amount": "成交额",
    })
    df["时间"] = pd.to_datetime(df["时间"])
    df = df[df["时间"].dt.strftime("%Y-%m-%d") == date_str]
    df = df.sort_values("时间").reset_index(drop=True)
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_min_kline_em(code: str, date_str: str) -> pd.DataFrame:
    start = f"{date_str} 09:30:00"
    end = f"{date_str} 15:00:00"
    return ak.stock_zh_a_hist_min_em(
        symbol=code, period="30", start_date=start, end_date=end, adjust="qfq"
    )


def fetch_min_kline(code: str, date_str: str) -> pd.DataFrame:
    # 快速重试：新浪优先
    for attempt in range(2):
        try:
            df = fetch_min_kline_sina(code, date_str)
            if not df.empty:
                print(f"  [source] 新浪：{len(df)} 根 30 分钟 K 线")
                return df
            if attempt == 0:
                print("  [source] 新浪无数据，尝试东方财富...")
        except Exception as e:
            if attempt == 0:
                print(f"  [source] 新浪失败({type(e).__name__})，尝试东方财富...")

    # 东方财富
    try:
        df = fetch_min_kline_em(code, date_str)
        if not df.empty:
            print(f"  [source] 东方财富：{len(df)} 根 30 分钟 K 线")
            return df
    except Exception as e:
        print(f"  [source] 东方财富失败({type(e).__name__})")

    weekday = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    if weekday >= 5:
        hint = f"{date_str} 为周末，非交易日。"
    else:
        hint = f"{date_str} 可能是非交易日或超出数据覆盖范围。"
    raise ValueError(f"未获取到 {code} 在 {date_str} 的30分钟行情数据。{hint}")


def compute_buy_sell(min_df: pd.DataFrame) -> dict:
    if min_df.empty:
        return {
            "buy_volume": 0, "sell_volume": 0, "neutral_volume": 0,
            "total_volume": 0, "buy_ratio": 0.0, "sell_ratio": 0.0,
            "neutral_ratio": 0.0, "buy_sell_ratio": None, "verdict": "无数据",
        }

    buy_vol = sell_vol = neutral_vol = 0.0
    for _, row in min_df.iterrows():
        vol, op, cl = float(row["成交量"]), float(row["开盘"]), float(row["收盘"])
        if cl > op:
            buy_vol += vol
        elif cl < op:
            sell_vol += vol
        else:
            buy_vol += vol * 0.5
            sell_vol += vol * 0.5
            neutral_vol += vol

    total = buy_vol + sell_vol + neutral_vol
    buy_ratio = buy_vol / total if total else 0.0
    sell_ratio = sell_vol / total if total else 0.0
    neutral_ratio = neutral_vol / total if total else 0.0
    buy_sell_ratio = buy_vol / sell_vol if sell_vol else None

    two_thirds = 2.0 / 3.0
    if buy_ratio >= two_thirds:
        verdict = "看多"
    elif sell_ratio >= two_thirds:
        verdict = "看空"
    elif buy_sell_ratio is not None and 1.0 <= buy_sell_ratio <= 1.2:
        verdict = "中性"
    elif buy_sell_ratio is not None and abs(buy_sell_ratio - 1.0) < 0.3:
        verdict = "中性"
    elif buy_ratio > sell_ratio:
        verdict = "偏多"
    else:
        verdict = "偏空"

    return {
        "buy_volume": round(buy_vol, 0),
        "sell_volume": round(sell_vol, 0),
        "neutral_volume": round(neutral_vol, 0),
        "total_volume": round(total, 0),
        "buy_ratio": round(buy_ratio, 4),
        "sell_ratio": round(sell_ratio, 4),
        "neutral_ratio": round(neutral_ratio, 4),
        "buy_sell_ratio": round(buy_sell_ratio, 4) if buy_sell_ratio is not None else None,
        "verdict": verdict,
    }


def summarize_daily(min_df: pd.DataFrame) -> dict:
    if min_df.empty:
        return {}
    return {
        "open": float(min_df.iloc[0]["开盘"]),
        "close": float(min_df.iloc[-1]["收盘"]),
        "high": float(min_df["最高"].max()),
        "low": float(min_df["最低"].min()),
        "volume": float(min_df["成交量"].sum()),
        "amount": float(min_df["成交额"].sum()),
        "pct_change": round(
            (float(min_df.iloc[-1]["收盘"]) - float(min_df.iloc[0]["开盘"]))
            / float(min_df.iloc[0]["开盘"]) * 100, 2),
        "change": round(
            float(min_df.iloc[-1]["收盘"]) - float(min_df.iloc[0]["开盘"]), 4),
    }


def build_payload(company: str, date_str: str) -> dict:
    code, std_name = resolve_stock_code(company)
    print(f"[resolve] {company} -> {code} ({std_name})")

    min_df = fetch_min_kline(code, date_str)
    if min_df.empty:
        raise ValueError(f"未获取到 {std_name}({code}) 在 {date_str} 的行情数据。")

    kline = []
    for _, row in min_df.iterrows():
        kline.append({
            "time": pd.Timestamp(row["时间"]).strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(row["开盘"]), "close": float(row["收盘"]),
            "high": float(row["最高"]), "low": float(row["最低"]),
            "volume": float(row["成交量"]), "amount": float(row["成交额"]),
        })

    return {
        "company": std_name,
        "company_input": company,
        "stock_code": code,
        "date": date_str,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "daily": summarize_daily(min_df),
        "kline_30min": kline,
        "buy_sell": compute_buy_sell(min_df),
    }


def save_json(payload: dict) -> Path:
    fname = f"{safe_filename(payload['company'])}_{payload['date']}.json"
    path = JSON_DATA_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[save_json] {path}")
    return path


def generate_html(payload: dict) -> Path:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(payload, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")
    html = template.replace("__DATA_JSON__", data_json)
    title = f"{payload.get('company','')} {payload.get('date','')} · 股票看板"
    html = html.replace("__PAGE_TITLE__", title)

    fname = f"{safe_filename(payload['company'])}_{payload['date']}.html"
    path = HTML_DATA_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[save_html] {path}")
    return path


def execute(company: str, date_str: str, from_json: bool = False, skip_html: bool = False) -> dict:
    """
    优化版核心执行函数 - 可被主进程直接import调用，避免subprocess开销
    
    返回结构化摘要（降低token消耗）
    """
    start_time = time.time()

    if from_json:
        fname = f"{safe_filename(company)}_{date_str}.json"
        path = JSON_DATA_DIR / fname
        if not path.exists():
            return {"status": "error", "message": f"json_data/下未找到 {fname}"}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = build_payload(company, date_str)

    json_path = save_json(payload)

    html_path = None
    if not skip_html:
        html_path = generate_html(payload)

    elapsed = time.time() - start_time
    bs = payload["buy_sell"]

    return {
        "status": "completed",
        "company": payload["company"],
        "stock_code": payload["stock_code"],
        "date": payload["date"],
        "kline_count": len(payload["kline_30min"]),
        "verdict": bs["verdict"],
        "buy_ratio": bs["buy_ratio"],
        "sell_ratio": bs["sell_ratio"],
        "buy_sell_ratio": bs["buy_sell_ratio"],
        "pct_change": payload["daily"].get("pct_change"),
        "json_path": str(json_path),
        "html_path": str(html_path) if html_path else None,
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="优化版股票信息获取")
    parser.add_argument("--company", "-c", required=True, help="公司名称")
    parser.add_argument("--date", "-d", required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--from-json", action="store_true", help="从已有JSON生成HTML")
    parser.add_argument("--skip-html", action="store_true", help="跳过HTML生成")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("日期格式错误，使用 YYYY-MM-DD")

    result = execute(args.company, args.date, args.from_json, args.skip_html)

    if result["status"] == "error":
        print(f"\n❌ {result['message']}")
        raise SystemExit(1)

    print(f"\n✅ 完成 ({result['elapsed_seconds']}s)")
    print(f"   公司: {result['company']} ({result['stock_code']})")
    print(f"   日期: {result['date']}")
    print(f"   K线: {result['kline_count']} 根")
    print(f"   判定: {result['verdict']}")
    print(f"   JSON: {result['json_path']}")
    if result.get("html_path"):
        print(f"   HTML: {result['html_path']}")


if __name__ == "__main__":
    main()