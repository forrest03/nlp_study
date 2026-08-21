# -*- coding: utf-8 -*-
"""股票查询模块 —— 供 harness 通过 importlib 动态加载调用。

导出（唯一公共接口）:
    run(query="") -> dict
        query: 公司名称或股票代码（如 腾讯 / Apple / 600519）
        成功时返回: symbol / name / display / exchange / currency / points（近 7 个交易日收盘价）
"""
import json
import sys

import requests

SEARCH_API = "https://smartbox.gtimg.cn/s3/"
KLINE_API = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
EXCHANGE_NAMES = {"sh": "沪市", "sz": "深市", "hk": "香港", "us": "美股"}
MARKET_CURRENCY = {"sh": "CNY", "sz": "CNY", "hk": "HKD", "us": "USD"}


def _search_stock(query):
    """通过腾讯智能搜索把公司名称解析为股票代码，优先返回股票型（GP）匹配。"""
    import re
    try:
        r = requests.get(
            SEARCH_API,
            params={"v": 2, "q": query, "t": "all"},
            headers=HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return None
        # 接口返回 v_hint="..." 形式的文本，值内可能含 \uXXXX 转义，按 JSON 字符串解码
        m = re.search(r'v_hint="(.*)"', r.text, re.S)
        if not m:
            return None
        hint = json.loads('"' + m.group(1) + '"')
    except Exception as e:  # noqa: BLE001
        print(f"[search error] {e}")
        return None

    for item in hint.split("^"):
        parts = item.split("~")
        if len(parts) < 5 or parts[4] != "GP":  # GP = 股票
            continue
        market, symbol, name = parts[0], parts[1], parts[2]
        if not symbol:
            continue
        display = symbol.split(".")[0].upper() if market == "us" else symbol
        return {
            "symbol": market + symbol,  # 腾讯 K 线接口使用的代码（如 sh600519 / usaapl.oq）
            "name": name or query,
            "display": display,
            "exchange": EXCHANGE_NAMES.get(market, market),
            "currency": MARKET_CURRENCY.get(market, ""),
        }
    return None


def _get_stock_history(code):
    """获取最近一周（7 个交易日）的日 K 线收盘价。"""
    try:
        r = requests.get(
            KLINE_API,
            params={"param": f"{code},day,,,7,qfq"},
            headers=HEADERS,
            timeout=12,
        )
        data = r.json() if r.status_code == 200 else None
    except Exception as e:  # noqa: BLE001
        print(f"[kline error] {e}")
        return []
    if not data or data.get("code") != 0:
        return []
    node = (data.get("data") or {}).get(code) or {}
    rows = node.get("qfqday") or node.get("day") or []
    # 每行格式：[日期, 开盘, 收盘, 最高, 最低, 成交量]
    return [
        {"date": row[0], "price": round(float(row[2]), 2)}
        for row in rows
        if len(row) >= 3
    ]


# ---------- 唯一公共接口 ----------

def run(query="", **kwargs):
    """查询公司近一周（7 个交易日）股票行情，返回 dict。"""
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "公司名称不能为空，示例: query='腾讯'"}
    info = _search_stock(query)
    if not info:
        return {
            "success": False,
            "error": f"未找到与 '{query}' 相关的股票，请尝试输入股票代码（如 AAPL、600519、0700）",
        }
    info["points"] = _get_stock_history(info["symbol"])
    if not info["points"]:
        return {"success": False, "error": f"获取 {info['name']} 行情失败，请稍后重试"}
    info["success"] = True
    return info


if __name__ == "__main__":
    _query = sys.argv[1] if len(sys.argv) > 1 else input("公司名称: ").strip()
    result = run(query=_query)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("success") else 1)
