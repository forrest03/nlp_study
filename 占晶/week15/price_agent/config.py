from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Platform:
    key: str
    label: str
    domains: tuple[str, ...]


PLATFORMS: tuple[Platform, ...] = (
    Platform("jd", "京东", ("jd.com", "3.cn")),
    Platform("taobao", "淘宝", ("taobao.com", "tmall.com")),
    Platform("pinduoduo", "拼多多", ("pinduoduo.com", "yangkeduo.com")),
)

MIN_CONFIDENCE = Decimal("0.65")
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REGION = "上海"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_OFFERS = 3
DEFAULT_MAX_REACT_STEPS = 2
MAX_REACT_STEPS = 3

