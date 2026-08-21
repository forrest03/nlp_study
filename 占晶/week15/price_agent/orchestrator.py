from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from .agents import demo_worker, live_worker
from .config import (
    DEFAULT_MAX_OFFERS,
    DEFAULT_MAX_REACT_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_REACT_STEPS,
    PLATFORMS,
    Platform,
)
from .domain import rank_offers, utc_now


async def compare_prices(
    product: str,
    variant: str,
    region: str,
    *,
    mode: str = "demo",
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_offers: int = DEFAULT_MAX_OFFERS,
    max_react_steps: int = DEFAULT_MAX_REACT_STEPS,
) -> dict[str, Any]:
    product, variant, region = _validate_query(
        product,
        variant,
        region,
        mode,
        max_offers,
        max_react_steps,
    )
    client = _create_live_client() if mode == "live" else None
    started_at = utc_now()
    started = time.perf_counter()

    tasks = [
        asyncio.create_task(
            _run_platform(
                platform,
                client,
                product,
                variant,
                region,
                mode,
                model,
                timeout,
                max_offers,
                max_react_steps,
            ),
            name=platform.key,
        )
        for platform in PLATFORMS
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    platform_results = [
        _error_result(platform, outcome) if isinstance(outcome, BaseException) else outcome
        for platform, outcome in zip(PLATFORMS, outcomes, strict=True)
    ]

    succeeded = sum(result["status"] != "error" for result in platform_results)
    return {
        "query": {
            "product": product,
            "variant": variant,
            "region": region,
            "platforms": [platform.label for platform in PLATFORMS],
        },
        "run": {
            "mode": mode,
            "model": model if mode == "live" else None,
            "started_at": started_at,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "agents_total": len(PLATFORMS),
            "agents_succeeded": succeeded,
            "agents_failed": len(PLATFORMS) - succeeded,
            "max_react_steps_per_agent": max_react_steps,
            "react_steps_used": sum(
                int(result.get("react_steps_used", 0)) for result in platform_results
            ),
        },
        "platform_results": platform_results,
        "ranked_offers": rank_offers(platform_results),
    }


def _validate_query(
    product: str,
    variant: str,
    region: str,
    mode: str,
    max_offers: int,
    max_react_steps: int,
) -> tuple[str, str, str]:
    product = product.strip()
    variant = variant.strip()
    region = region.strip()
    if not product:
        raise ValueError("商品名称不能为空")
    if not variant:
        raise ValueError("必须提供明确规格，避免比较不同 SKU")
    if not region:
        raise ValueError("收货地区不能为空")
    if mode not in {"demo", "live"}:
        raise ValueError("mode 必须是 demo 或 live")
    if max_offers < 1 or max_offers > 10:
        raise ValueError("max_offers 必须在 1 到 10 之间")
    if max_react_steps < 1 or max_react_steps > MAX_REACT_STEPS:
        raise ValueError(f"max_react_steps 必须在 1 到 {MAX_REACT_STEPS} 之间")
    return product, variant, region


def _create_live_client() -> Any:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Live 模式需要先设置 OPENAI_API_KEY")
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 包，请运行：python -m pip install -r requirements.txt") from exc
    return AsyncOpenAI()


async def _run_platform(
    platform: Platform,
    client: Any,
    product: str,
    variant: str,
    region: str,
    mode: str,
    model: str,
    timeout: float,
    max_offers: int,
    max_react_steps: int,
) -> dict[str, Any]:
    if mode == "demo":
        return await demo_worker(platform, product, variant, region)
    return await live_worker(
        client,
        platform,
        product,
        variant,
        region,
        model,
        timeout,
        max_offers,
        max_react_steps,
    )


def _error_result(platform: Platform, error: BaseException) -> dict[str, Any]:
    return {
        "platform": platform.label,
        "status": "error",
        "warning": f"{type(error).__name__}: {error}",
        "offers": [],
        "react_steps_used": 0,
        "stop_reason": "stop_error",
        "react_trace": [],
    }

