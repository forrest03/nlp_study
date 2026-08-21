# -*- coding: utf-8 -*-
"""
获取股票信息脚本（新架构）

工作流：
    1. 使用 akshare 拉取指定公司+日期的 30 分钟 K 线；
    2. 计算买卖成交量与多空判定；
    3. 保存 JSON 到 json_data/{公司}_{日期}.json；
    4. 基于 scripts/dashboard_template.html 生成独立 HTML 页面，
       保存到 html_data/{公司}_{日期}.html，可直接双击打开。

目录约定：
    本脚本位于 scripts/ 目录，根目录结构：
    stock-dashboard/
    ├── json_data/      ← JSON 原始数据
    ├── html_data/      ← 生成的 HTML 页面
    ├── scripts/        ← 本脚本与模板
    ├── static/         ← 共享静态资源 (echarts.min.js)
    └── SKILL.md
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from functools import wraps
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

# ---- 目录定位（相对本脚本位置）----
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent                 # stock-dashboard/
JSON_DATA_DIR = ROOT_DIR / "json_data"       # 存放 JSON
HTML_DATA_DIR = ROOT_DIR / "html_data"       # 存放 HTML 页面
TEMPLATE_PATH = SCRIPT_DIR / "dashboard_template.html"

for d in (JSON_DATA_DIR, HTML_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 重试装饰器
# ---------------------------------------------------------------------------
def with_retries(retries: int = 4, backoff: float = 1.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError, TimeoutError, OSError) as e:
                    last_err = e
                    wait = backoff * (i + 1)
                    print(f"  [retry] {func.__name__} 第 {i+1}/{retries} 次失败："
                          f"{type(e).__name__}，{wait:.1f}s 后重试...")
                    time.sleep(wait)
            raise last_err
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", "", text).lower()


def safe_filename(text: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", text).strip("_")


# ---------------------------------------------------------------------------
# 公司名 -> 股票代码
# ---------------------------------------------------------------------------
@with_retries()
def resolve_stock_code(company: str) -> tuple[str, str]:
    df = ak.stock_info_a_code_name()
    target = _normalize(company)

    mask = df["name"].apply(_normalize) == target
    if mask.any():
        row = df[mask].iloc[0]
        return str(row["code"]), str(row["name"])

    contains = df["name"].apply(lambda n: target in _normalize(n))
    if contains.any():
        row = df[contains].iloc[0]
        return str(row["code"]), str(row["name"])

    rev = df["name"].apply(lambda n: _normalize(n) in target)
    if rev.any():
        row = df[rev].iloc[0]
        return str(row["code"]), str(row["name"])

    raise ValueError(f"未找到匹配的股票：{company}")


def market_prefix(code: str) -> str:
    c = code.strip()
    if c.startswith("6"): return "sh"
    if c.startswith(("0", "3")): return "sz"
    if c.startswith(("8", "4")): return "bj"
    return "sz"


# ---------------------------------------------------------------------------
# 数据获取
# ---------------------------------------------------------------------------
@with_retries()
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


@with_retries()
def fetch_min_kline_em(code: str, date_str: str) -> pd.DataFrame:
    start = f"{date_str} 09:30:00"
    end = f"{date_str} 15:00:00"
    df = ak.stock_zh_a_hist_min_em(
        symbol=code, period="30", start_date=start, end_date=end, adjust="qfq"
    )
    # 统一列名，与新浪源保持一致
    col_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if '时间' in cl or 'time' in cl: col_map[c] = '时间'
        elif '开盘' in cl or 'open' in cl: col_map[c] = '开盘'
        elif '最高' in cl or 'high' in cl: col_map[c] = '最高'
        elif '最低' in cl or 'low' in cl: col_map[c] = '最低'
        elif '收盘' in cl or 'close' in cl: col_map[c] = '收盘'
        elif '成交' in cl and '量' in cl: col_map[c] = '成交量'
        elif '成交' in cl and '额' in cl: col_map[c] = '成交额'
    df = df.rename(columns=col_map)
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "时间" in df.columns:
        df["时间"] = pd.to_datetime(df["时间"])
        df = df.sort_values("时间").reset_index(drop=True)
    return df


def fetch_min_kline(code: str, date_str: str) -> pd.DataFrame:
    # 1. 新浪
    try:
        df = fetch_min_kline_sina(code, date_str)
        if not df.empty:
            print(f"  [source] 新浪：{len(df)} 根 30 分钟 K 线")
            return df
        print("  [source] 新浪无数据，尝试东方财富...")
    except Exception as e:
        print(f"  [source] 新浪失败({type(e).__name__})，尝试东方财富...")

    # 2. 东方财富
    try:
        df = fetch_min_kline_em(code, date_str)
        if not df.empty:
            print(f"  [source] 东方财富：{len(df)} 根 30 分钟 K 线")
            return df
    except Exception as e:
        print(f"  [source] 东方财富失败({type(e).__name__})")

    # 3. 都没数据
    weekday = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    if weekday >= 5:
        hint = f"{date_str} 为周末，非交易日。"
    else:
        hint = (f"{date_str} 可能是非交易日、节假日，或超出数据源覆盖范围"
                f"（新浪约覆盖近 1 年）。")
    raise ValueError(
        f"未获取到 {code} 在 {date_str} 的 30 分钟行情数据。{hint}"
        "请更换为有效交易日后重试。"
    )


# ---------------------------------------------------------------------------
# 买入/卖出成交量推算
# ---------------------------------------------------------------------------
def compute_buy_sell(min_df: pd.DataFrame) -> dict:
    if min_df.empty:
        return {
            "buy_volume": 0, "sell_volume": 0, "neutral_volume": 0,
            "total_volume": 0, "buy_ratio": 0.0, "sell_ratio": 0.0,
            "neutral_ratio": 0.0, "buy_sell_ratio": None, "verdict": "无数据",
        }

    buy_vol = sell_vol = neutral_vol = 0.0
    for _, row in min_df.iterrows():
        try:
            vol = float(row.get("成交量", 0) or 0)
            op = float(row.get("开盘", 0) or 0)
            cl = float(row.get("收盘", 0) or 0)
        except (TypeError, ValueError):
            continue
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

    def safe_val(v, default=0.0):
        try:
            if v is None:
                return default
            if isinstance(v, float) and v != v:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    first = min_df.iloc[0]
    last = min_df.iloc[-1]

    open_val = safe_val(first.get("开盘"))
    close_val = safe_val(last.get("收盘"))

    high_val = 0.0
    low_val = 0.0
    vol_val = 0.0
    amt_val = 0.0
    try:
        if "最高" in min_df.columns:
            high_val = safe_val(min_df["最高"].max())
    except Exception:
        pass
    try:
        if "最低" in min_df.columns:
            low_val = safe_val(min_df["最低"].min())
    except Exception:
        pass
    try:
        if "成交量" in min_df.columns:
            vol_val = safe_val(min_df["成交量"].sum())
    except Exception:
        pass
    try:
        if "成交额" in min_df.columns:
            amt_val = safe_val(min_df["成交额"].sum())
    except Exception:
        pass

    return {
        "open": open_val,
        "close": close_val,
        "high": high_val,
        "low": low_val,
        "volume": vol_val,
        "amount": amt_val,
        "pct_change": round(
            (close_val - open_val) / open_val * 100 if open_val else 0, 2),
        "change": round(close_val - open_val, 4),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_payload(company: str, date_str: str) -> dict:
    code, std_name = resolve_stock_code(company)
    print(f"[resolve] 公司='{company}' -> 代码={code}  标准名称={std_name}")

    min_df = fetch_min_kline(code, date_str)
    if min_df.empty:
        raise ValueError(f"未获取到 {std_name}({code}) 在 {date_str} 的行情数据。")

    kline = []
    for _, row in min_df.iterrows():
        def _safe_float(key, default=0.0):
            try:
                v = row.get(key)
                if v is None:
                    return default
                if isinstance(v, float) and v != v:
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        try:
            time_str = pd.Timestamp(row["时间"]).strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError, KeyError):
            time_str = str(row.get("时间", ""))

        kline.append({
            "time": time_str,
            "open": _safe_float("开盘"), "close": _safe_float("收盘"),
            "high": _safe_float("最高"), "low": _safe_float("最低"),
            "volume": _safe_float("成交量"), "amount": _safe_float("成交额"),
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
    """基于模板生成独立 HTML 页面到 html_data/。"""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(payload, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")  # 防止 script 标签被截断
    html = template.replace("__DATA_JSON__", data_json)
    title = f"{payload.get('company','')} {payload.get('date','')} · 股票看板"
    html = html.replace("__PAGE_TITLE__", title)

    fname = f"{safe_filename(payload['company'])}_{payload['date']}.html"
    path = HTML_DATA_DIR / fname
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[save_html] {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="获取股票信息 → 生成 JSON + HTML 页面"
    )
    parser.add_argument("--company", "-c", required=True, help="公司名称，如 平安银行")
    parser.add_argument("--date", "-d", required=True,
                        help="日期 YYYY-MM-DD，如 2026-07-28")
    parser.add_argument("--from-json", action="store_true",
                        help="从 json_data/ 已有 JSON 生成 HTML，不联网获取")
    parser.add_argument("--skip-html", action="store_true",
                        help="仅生成 JSON，跳过 HTML")
    parser.add_argument("--skip-json", action="store_true",
                        help="仅生成 HTML，跳过 JSON（需已有对应 JSON 或联网拉取）")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        raise SystemExit("日期格式错误，请使用 YYYY-MM-DD。")

    if args.from_json:
        # 从 json_data/ 读取已有数据
        fname = f"{safe_filename(args.company)}_{args.date}.json"
        path = JSON_DATA_DIR / fname
        if not path.exists():
            raise SystemExit(f"json_data/ 下未找到 {fname}，请先运行不带 --from-json 的命令获取数据。")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"[from_json] 读取已有数据：{path}")
    else:
        payload = build_payload(args.company, args.date)

    if not args.skip_json:
        save_json(payload)
    if not args.skip_html:
        generate_html(payload)

    bs = payload["buy_sell"]
    print("\n========== 摘要 ==========")
    print(f"公司: {payload['company']} ({payload['stock_code']})")
    print(f"日期: {payload['date']}")
    print(f"30分钟K线: {len(payload['kline_30min'])} 根")
    print(f"买入占比: {bs['buy_ratio']:.2%}  卖出占比: {bs['sell_ratio']:.2%}")
    ratio_str = f"{bs['buy_sell_ratio']:.4f}" if bs['buy_sell_ratio'] is not None else "N/A"
    print(f"买入/卖出比: {ratio_str}")
    print(f"判定结果: {bs['verdict']}")
    print("=========================\n")
    print("查看方式：")
    print(f"  1. 直接双击打开：html_data/{safe_filename(payload['company'])}_{payload['date']}.html")
    print(f"  2. 启动服务：cd scripts && uvicorn app:app --reload")
    print(f"     访问：http://127.0.0.1:8000/stock/{payload['company']}/{payload['date']}")


if __name__ == "__main__":
    main()
