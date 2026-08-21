from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse

from .config import MIN_CONFIDENCE, Platform


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def to_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number < 0:
        return None
    return number.quantize(Decimal("0.01"))


def decimal_to_json(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def host_is_allowed(url: str, domains: tuple[str, ...]) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def normalize_offer(raw: dict[str, Any], platform: Platform) -> dict[str, Any] | None:
    url = str(raw.get("url", "")).strip()
    if not host_is_allowed(url, platform.domains):
        return None

    listed = to_decimal(raw.get("listed_price"))
    discount = to_decimal(raw.get("immediate_discount"))
    shipping = to_decimal(raw.get("shipping"))
    reported_final = to_decimal(raw.get("final_price"))

    note = str(raw.get("price_note", "")).strip()
    if listed is not None and discount is not None and shipping is not None:
        if discount > listed:
            discount = Decimal("0.00")
            note = f"{note}；即时优惠大于标价，已忽略".strip("；")
        computed = (listed - discount + shipping).quantize(Decimal("0.01"))
        if reported_final is not None and abs(computed - reported_final) > Decimal("0.01"):
            note = f"{note}；模型报告价与组成项不一致，已按组成项重算".strip("；")
        final_price = computed
    else:
        final_price = reported_final

    if final_price is None or final_price <= 0:
        return None

    confidence = to_decimal(raw.get("match_confidence")) or Decimal("0.00")
    confidence = min(confidence, Decimal("1.00"))

    return {
        "platform": platform.label,
        "product_name": str(raw.get("product_name", "")).strip(),
        "variant": str(raw.get("variant", "")).strip(),
        "condition": str(raw.get("condition", "unknown")),
        "seller": str(raw.get("seller", "")).strip(),
        "listed_price": decimal_to_json(listed),
        "immediate_discount": decimal_to_json(discount),
        "shipping": decimal_to_json(shipping),
        "final_price": decimal_to_json(final_price),
        "currency": "CNY",
        "price_note": note,
        "url": url,
        "source_title": str(raw.get("source_title", "")).strip(),
        "collected_at": str(raw.get("collected_at", "")).strip() or utc_now(),
        "match_confidence": float(confidence),
        "match_reason": str(raw.get("match_reason", "")).strip(),
        "exact_variant": bool(raw.get("exact_variant", False)),
    }


def normalize_platform_result(raw: dict[str, Any], platform: Platform) -> dict[str, Any]:
    offers: list[dict[str, Any]] = []
    rejected = 0
    for item in raw.get("offers", []):
        if not isinstance(item, dict):
            rejected += 1
            continue
        offer = normalize_offer(item, platform)
        if offer is None:
            rejected += 1
        else:
            offers.append(offer)

    warning = str(raw.get("warning", "")).strip()
    if rejected:
        warning = f"{warning}；本地校验剔除 {rejected} 条无效结果".strip("；")
    return {
        "platform": platform.label,
        "status": "ok" if offers else "no_result",
        "warning": warning,
        "search_summary": str(raw.get("search_summary", "")).strip(),
        "next_search_focus": str(raw.get("next_search_focus", "")).strip(),
        "offers": offers,
    }


def offer_is_rankable(offer: dict[str, Any]) -> bool:
    if offer.get("condition") != "new":
        return False
    if not offer.get("exact_variant"):
        return False
    if Decimal(str(offer.get("match_confidence", 0))) < MIN_CONFIDENCE:
        return False
    price = to_decimal(offer.get("final_price"))
    return price is not None and price > 0


def rank_offers(platform_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, float]] = set()

    for result in platform_results:
        for offer in result.get("offers", []):
            if not offer_is_rankable(offer):
                continue
            price = offer["final_price"]
            key = (
                str(offer.get("platform", "")),
                str(offer.get("seller", "")).casefold(),
                str(offer.get("product_name", "")).casefold(),
                float(price),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(offer)

    return sorted(
        candidates,
        key=lambda item: (
            float(item["final_price"]),
            -float(item["match_confidence"]),
            item["platform"],
        ),
    )


def merge_offers(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {
        (offer.get("url"), offer.get("variant"), offer.get("final_price"))
        for offer in existing
    }
    for offer in incoming:
        key = (offer.get("url"), offer.get("variant"), offer.get("final_price"))
        if key not in seen:
            seen.add(key)
            merged.append(offer)
    return merged

