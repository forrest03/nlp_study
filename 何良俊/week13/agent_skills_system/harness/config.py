"""
config — harness 配置加载
============================

读取顺序（前者优先）：
  1. 环境变量 DEEPSEEK_API_KEY / DEEPSEEK_MODEL / DEEPSEEK_BASE_URL
  2. 项目根目录下的 .env 文件（KEY=VALUE 行，#开头为注释）
  3. 内置默认值

不依赖 python-dotenv，自己实现一个最小的 .env 解析器以保持零依赖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_dotenv(path: Path) -> dict[str, str]:
    """最小化的 .env 解析：KEY=VALUE，忽略注释和空行，VALUE 支持去引号。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        out[k] = v
    return out


@dataclass
class HarnessConfig:
    """运行时配置。"""

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"
    request_timeout: int = 120
    llm_temperature: float = 0.0

    @classmethod
    def load(cls, dotenv_path: Optional[Path] = None) -> "HarnessConfig":
        dotenv_path = dotenv_path or _PROJECT_ROOT / ".env"
        file_env = _parse_dotenv(dotenv_path)

        def _get(key: str, default: str = "") -> str:
            # 环境变量优先于 .env 文件
            return os.environ.get(key) or file_env.get(key) or default

        return cls(
            deepseek_api_key=_get("DEEPSEEK_API_KEY"),
            deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            request_timeout=int(_get("DEEPSEEK_TIMEOUT", "120")),
            llm_temperature=float(_get("DEEPSEEK_TEMPERATURE", "0.0")),
        )

    @property
    def llm_available(self) -> bool:
        return bool(self.deepseek_api_key)
