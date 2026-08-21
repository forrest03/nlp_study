from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from .config import (
    DEFAULT_MAX_OFFERS,
    DEFAULT_MAX_REACT_STEPS,
    DEFAULT_MODEL,
    DEFAULT_REGION,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_REACT_STEPS,
)
from .orchestrator import compare_prices


def money(value: Any) -> str:
    if value is None:
        return "未知"
    return f"￥{float(value):,.2f}"


def render_markdown(result: dict[str, Any]) -> str:
    query = result["query"]
    run = result["run"]
    lines = [
        f"# {query['product']} 比价结果",
        "",
        f"- 规格：{query['variant']}",
        f"- 地区：{query['region']}",
        f"- 模式：{run['mode']}",
        f"- 并行 Agent：{run['agents_succeeded']}/{run['agents_total']} 成功",
        f"- ReAct 步骤：共 {run['react_steps_used']} 步（每个平台最多 {run['max_react_steps_per_agent']} 步）",
        f"- 用时：{run['elapsed_seconds']:.3f} 秒",
        "",
    ]
    _append_offers(lines, result["ranked_offers"])
    _append_warnings(lines, result["platform_results"])
    _append_react_trace(lines, result["platform_results"])
    lines.append("价格会随账号、地区、活动和库存变化；下单前请打开来源页面确认最终结算价。")
    return "\n".join(lines)


def _append_offers(lines: list[str], offers: list[dict[str, Any]]) -> None:
    if not offers:
        lines.extend(["没有找到通过规格和价格校验的报价。", ""])
        return

    lines.extend(
        [
            "| 排名 | 平台 | 到手价 | 标价 | 商品与规格 | 店铺 | 置信度 |",
            "|---:|---|---:|---:|---|---|---:|",
        ]
    )
    for index, offer in enumerate(offers, start=1):
        product_variant = f"{offer['product_name']} / {offer['variant']}".replace("|", "\\|")
        seller = offer["seller"].replace("|", "\\|")
        lines.append(
            f"| {index} | {offer['platform']} | {money(offer['final_price'])} | "
            f"{money(offer['listed_price'])} | [{product_variant}]({offer['url']}) | "
            f"{seller} | {offer['match_confidence']:.0%} |"
        )
    lines.append("")


def _append_warnings(lines: list[str], platform_results: list[dict[str, Any]]) -> None:
    warnings = [
        f"- {item['platform']}：{item['warning']}"
        for item in platform_results
        if item.get("warning")
    ]
    if warnings:
        lines.extend(["## 提示", "", *warnings, ""])


def _append_react_trace(lines: list[str], platform_results: list[dict[str, Any]]) -> None:
    lines.extend(["## Subagent ReAct 轨迹", ""])
    for item in platform_results:
        lines.append(
            f"- {item['platform']}：{item.get('react_steps_used', 0)} 步，"
            f"停止原因 `{item.get('stop_reason', 'unknown')}`"
        )
        for step in item.get("react_trace", []):
            lines.append(
                f"  - Step {step['step']}：{step['action']} → {step['observation']} "
                f"→ `{step['decision']}`"
            )
    lines.append("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="京东、淘宝、拼多多并行比价 Agent")
    parser.add_argument("product", help="商品名称，例如：iPhone 17 Pro")
    parser.add_argument("--variant", required=True, help="明确规格，例如：256GB 黑色 全新国行")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"收货地区，默认：{DEFAULT_REGION}")
    parser.add_argument("--mode", choices=("demo", "live"), default="demo")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="单个平台超时秒数")
    parser.add_argument("--max-offers", type=int, default=DEFAULT_MAX_OFFERS, help="每个平台最多返回结果数")
    parser.add_argument(
        "--max-react-steps",
        type=int,
        default=DEFAULT_MAX_REACT_STEPS,
        choices=range(1, MAX_REACT_STEPS + 1),
        help=f"每个平台最多 Observe-Act 步数，默认：{DEFAULT_MAX_REACT_STEPS}",
    )
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = asyncio.run(
            compare_prices(
                args.product,
                args.variant,
                args.region,
                mode=args.mode,
                model=args.model,
                timeout=args.timeout,
                max_offers=args.max_offers,
                max_react_steps=args.max_react_steps,
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"错误：{exc}") from exc

    if args.json:
        # ASCII-safe JSON avoids Windows PowerShell 5.1 corrupting UTF-8 text in a pipe.
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print(render_markdown(result))

