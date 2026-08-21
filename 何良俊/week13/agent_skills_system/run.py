"""项目入口：python run.py [args]  等价于 python -m harness.cli [args]"""
from harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
