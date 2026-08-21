from __future__ import annotations

from typing import Any


OFFER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "platform": {"type": "string"},
        "product_name": {"type": "string"},
        "variant": {"type": "string"},
        "condition": {
            "type": "string",
            "enum": ["new", "used", "refurbished", "unknown"],
        },
        "seller": {"type": "string"},
        "listed_price": {"type": ["number", "null"]},
        "immediate_discount": {"type": ["number", "null"]},
        "shipping": {"type": ["number", "null"]},
        "final_price": {"type": ["number", "null"]},
        "currency": {"type": "string", "enum": ["CNY"]},
        "price_note": {"type": "string"},
        "url": {"type": "string"},
        "source_title": {"type": "string"},
        "collected_at": {"type": "string"},
        "match_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "match_reason": {"type": "string"},
        "exact_variant": {"type": "boolean"},
    },
    "required": [
        "platform",
        "product_name",
        "variant",
        "condition",
        "seller",
        "listed_price",
        "immediate_discount",
        "shipping",
        "final_price",
        "currency",
        "price_note",
        "url",
        "source_title",
        "collected_at",
        "match_confidence",
        "match_reason",
        "exact_variant",
    ],
    "additionalProperties": False,
}


PLATFORM_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "platform": {"type": "string"},
        "status": {"type": "string", "enum": ["ok", "no_result"]},
        "warning": {"type": "string"},
        "search_summary": {"type": "string"},
        "next_search_focus": {"type": "string"},
        "offers": {"type": "array", "items": OFFER_SCHEMA},
    },
    "required": [
        "platform",
        "status",
        "warning",
        "search_summary",
        "next_search_focus",
        "offers",
    ],
    "additionalProperties": False,
}

