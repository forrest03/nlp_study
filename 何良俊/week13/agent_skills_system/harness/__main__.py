"""支持 `python -m harness` 入口。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
