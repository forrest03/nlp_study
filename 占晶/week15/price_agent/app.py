"""Backward-compatible application facade.

Implementation lives in focused modules; this file intentionally exposes only the
public entry points used by ``run.py`` and existing imports.
"""

from .cli import main
from .orchestrator import compare_prices

__all__ = ["compare_prices", "main"]


if __name__ == "__main__":
    main()

