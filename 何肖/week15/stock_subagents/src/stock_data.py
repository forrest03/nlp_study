"""股票数据获取封装（复用 stock-dashboard-optimized 的 akshare 拉取逻辑）

教学定位：
  本项目的核心是「主 agent + 并行 subagent」编排，数据获取只是工具。
  这里把 stock-dashboard-optimized/scripts/fetch_stock_opt.py 的核心逻辑抽出来，
  提供 get_stock_data(company, date) -> 结构化字典 供主 agent 工具调用。
  为防止 LLM 数据幻觉，增加前 4 个交易日日线历史，格式化为严格的 Markdown 表格。

依赖：akshare + pandas
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
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


def resolve_stock_code(company: str) -> tuple[str, str]:
    cache = _load_code_cache()
    if not cache:
        cache = _build_code_cache()
    target = _normalize(company)

    if target in cache:
        e = cache[target]
        return e["code"], e["name"]
    for name_lower, entry in cache.items():
        if target in name_lower:
            return entry["code"], entry["name"]
    for name_lower, entry in cache.items():
        if name_lower in target:
            return entry["code"], entry["name"]
    cache = _build_code_cache()
    if target in cache:
        e = cache[target]
        return e["code"], e["name"]
    raise ValueError(f"未找到匹配的股票：{company}")


def market_prefix(code: str) -> str:
    c = code.strip()
    if c.startswith("6"):
        return "sh"
    if c.startswith(("0", "3")):
        return "sz"
    if c.startswith(("8", "4")):
        return "bj"
    return "sz"


def _fetch_min_kline_sina(code: str, date_str: str) -> pd.DataFrame:
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


def _fetch_min_kline_em(code: str, date_str: str) -> pd.DataFrame:
    start = f"{date_str} 09:30:00"
    end = f"{date_str} 15:00:00"
    return ak.stock_zh_a_hist_min_em(
        symbol=code, period="30", start_date=start, end_date=end, adjust="qfq"
    )


def fetch_min_kline(code: str, date_str: str) -> pd.DataFrame:
    for attempt in range(2):
        try:
            df = _fetch_min_kline_sina(code, date_str)
            if not df.empty:
                print(f"  [source] 新浪：{len(df)} 根 30 分钟 K 线")
                return df
            if attempt == 0:
                print("  [source] 新浪无数据，尝试东方财富...")
        except Exception as e:
            if attempt == 0:
                print(f"  [source] 新浪失败({type(e).__name__})，尝试东方财富...")
    try:
        df = _fetch_min_kline_em(code, date_str)
        if not df.empty:
            print(f"  [source] 东方财富：{len(df)} 根 30 分钟 K 线")
            return df
    except Exception as e:
        print(f"  [source] 东方财富失败({type(e).__name__})")

    weekday = datetime.strptime(date_str, "%Y-%m-%d").weekday()
    hint = f"{date_str} 为周末，非交易日。" if weekday >= 5 \
        else f"{date_str} 可能是非交易日或超出数据覆盖范围。"
    raise ValueError(f"未获取到 {code} 在 {date_str} 的30分钟行情数据。{hint}")


def _fetch_daily_history_em(code: str, start: str, end: str) -> pd.DataFrame:
    """东方财富日线（stock_zh_a_hist）。"""
    return ak.stock_zh_a_hist(symbol=code, period="daily",
                              start_date=start, end_date=end, adjust="qfq")


def _fetch_daily_history_sina(code: str, start: str, end: str) -> pd.DataFrame:
    """新浪日线（stock_zh_a_daily），需要带市场前缀的代码。

    code 形如 "002594"（深市）/"600000"（沪市）/"300001"（创业板）。
    规则：6/9 开头 → 沪市 sh，其余 → 深市 sz。
    """
    if not code or not code[0].isdigit():
        raise ValueError(f"非法股票代码: {code}")
    prefix = "sh" if code[0] in ("6", "9") else "sz"
    sina_symbol = f"{prefix}{code}"
    df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start, end_date=end, adjust="qfq")
    # 新浪接口列名与东方财富不同，需统一
    if not df.empty:
        rename_map = {
            "date": "date", "open": "open", "close": "close",
            "high": "high", "low": "low", "volume": "volume",
            "amount": "amount",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        # 新浪无 pct_change，自行计算
        if "close" in df.columns and "pct_change" not in df.columns:
            df["pct_change"] = df["close"].pct_change() * 100
    return df


def _fetch_daily_history(code: str, date_str: str, days: int = 5) -> list:
    """获取指定日期及之前若干个交易日的日线数据。

    策略：东方财富 → 新浪 双源重试，每个源重试 2 次，间隔 1 秒。
    返回 list[dict]（JSON 可序列化）。
    失败时返回空 list（不抛异常，避免阻塞主流程）。
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # 向前追溯 40 个自然日足够覆盖 5 个交易日
    start = (dt - timedelta(days=40)).strftime("%Y%m%d")
    end = dt.strftime("%Y%m%d")

    sources = [
        ("东方财富", _fetch_daily_history_em),
        ("新浪", _fetch_daily_history_sina),
    ]

    df = pd.DataFrame()
    for src_name, fetcher in sources:
        for attempt in range(2):
            try:
                df = fetcher(code, start, end)
                if df is not None and not df.empty:
                    print(f"  [daily] {src_name}：{len(df)} 条日线")
                    break
            except Exception as e:
                print(f"  [daily] {src_name}失败(尝试 {attempt+1}/2): {type(e).__name__}: {str(e)[:80]}")
                if attempt == 0:
                    time.sleep(1.0)
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        return []

    # 统一列名（新浪源已经改过名，这里再统一一次）
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "涨跌幅": "pct_change",
    })
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "amount", "pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 只保留最近 days 个交易日
    df = df.sort_values("date").tail(days).reset_index(drop=True)

    # 转为 list[dict]（JSON 可序列化）
    records = []
    for _, row in df.iterrows():
        rec = {"date": row["date"].strftime("%Y-%m-%d")}
        for col in ["open", "close", "high", "low", "pct_change",
                    "volume", "amount"]:
            if col in row.index:
                v = row[col]
                if pd.isna(v):
                    rec[col] = None
                elif col in ("volume", "amount"):
                    rec[col] = int(float(v))
                else:
                    rec[col] = round(float(v), 4)
        records.append(rec)
    return records


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


def get_stock_data(company: str, date_str: str) -> dict:
    """主 agent 工具：拉取某公司某日的 30 分钟 K 线 + 摘要 + 买卖盘判定 + 前4日日线。

    返回结构化 dict（含原始 kline_30min 与摘要 daily/buy_sell），
    供看多/看空 subagent 共享分析。
    附加 daily_history：最近 5 个交易日日线，用于多空分析师对比分析。
    """
    t0 = time.time()
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

    daily_hist_df = _fetch_daily_history(code, date_str, days=5)

    payload = {
        "company": std_name,
        "company_input": company,
        "stock_code": code,
        "date": date_str,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "daily": summarize_daily(min_df),
        "daily_history": daily_hist_df,
        "kline_30min": kline,
        "buy_sell": compute_buy_sell(min_df),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    return payload


def format_stock_brief(payload: dict) -> str:
    """把股票数据格式化成喂给 LLM 的紧凑文本 + 5日 Markdown 表格。

    关键设计：
    - 使用严格的 Markdown 表格呈现 5 个交易日的日线数据，防止 LLM 幻觉。
    - 表格是看多/看空 subagent 唯一的数据来源，禁止编造数据。
    """
    if not payload:
        return "无股票数据"
    d = payload.get("daily", {})
    bs = payload.get("buy_sell", {})
    klines = payload.get("kline_30min", [])
    daily_hist = payload.get("daily_history")

    parts = [
        f"# 股票分析数据（唯一数据源，严禁编造）",
        f"**公司**：{payload['company']}({payload['stock_code']})  **查询日期**：{payload['date']}",
        f"**数据拉取时间**：{payload.get('fetched_at', '')}",
        "",
    ]

    # 5 日日线表格（核心！防止 LLM 幻觉）
    if daily_hist:  # list[dict]
        parts.append("## 近 5 个交易日行情（表格中的数字是唯一真实数据）")
        parts.append("")
        # 表头（按固定列顺序）
        cols = [("date", "日期"), ("open", "开盘"), ("close", "收盘"),
                ("high", "最高"), ("low", "最低"), ("pct_change", "涨跌幅%"),
                ("volume", "成交量(手)"), ("amount", "成交额(元)")]
        # 只保留记录里实际存在的列
        present_cols = [(c, l) for c, l in cols if daily_hist and c in daily_hist[0]]
        header = "| " + " | ".join(label for _, label in present_cols) + " |"
        sep = "|" + "|".join(["---"] * len(present_cols)) + "|"
        parts.append(header)
        parts.append(sep)
        for rec in daily_hist:
            vals = []
            for col, _ in present_cols:
                v = rec.get(col)
                if v is None:
                    vals.append("-")
                elif col in ("volume", "amount"):
                    vals.append(f"{int(v):,}")
                elif col == "pct_change":
                    vals.append(f"{float(v):+.2f}")
                elif col == "date":
                    vals.append(str(v))
                else:
                    vals.append(f"{float(v):.2f}")
            parts.append("| " + " | ".join(vals) + " |")
        parts.append("")
    else:
        parts.append("> ⚠️ 未获取到历史日线数据")
        parts.append("")

    # 当日详情
    parts.append("## 当日详情（查询日）")
    parts.append(f"- 开盘: **{d.get('open')}** 收盘: **{d.get('close')}** 最高: **{d.get('high')}** 最低: **{d.get('low')}**")
    parts.append(f"- 涨跌幅: **{d.get('pct_change')}%** 涨跌额: **{d.get('change')}**")
    parts.append(f"- 成交量: **{int(d.get('volume', 0)):,}** 成交额: **{int(d.get('amount', 0)):,}**")
    parts.append("")

    # 买卖盘
    parts.append("## 当日买卖盘分析")
    parts.append(f"- 买入占比: **{bs.get('buy_ratio')}** 卖出占比: **{bs.get('sell_ratio')}**")
    parts.append(f"- 买卖比: **{bs.get('buy_sell_ratio')}** 机器判定: **{bs.get('verdict')}**")
    parts.append("")

    # 30 分钟 K 线（紧凑列表）
    if klines:
        parts.append(f"## 当日 30 分钟 K 线（共 {len(klines)} 根）")
        kline_str = "\n".join(
            f"- {k['time'][11:16]} 开:{k['open']} 收:{k['close']} 高:{k['high']} 低:{k['low']} 量:{int(k['volume']):,}"
            for k in klines
        )
        parts.append(kline_str)
        parts.append("")

    parts.append("---")
    parts.append("**🚫 重要约束**：上方表格中的数字是唯一可靠的数据来源，分析时必须引用表格中的具体数值，严禁编造任何数据！")
    return "\n".join(parts)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    p = get_stock_data("比亚迪", "2026-08-04")
    print(format_stock_brief(p)[:600])
