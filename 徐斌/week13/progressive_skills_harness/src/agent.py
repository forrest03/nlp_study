"""
CLI 入口：渐进式 Skills Harness

用法：
  python src/agent.py
  python src/agent.py -q "给我做张 resilient 的闪卡"
  python src/agent.py -q "统计这段文字的字数：今天天气不错" --once
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 保证可从任意 cwd 导入同目录模块
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from harness import run_and_print  # noqa: E402
from llm_config import current_model_info  # noqa: E402
from skill_registry import SkillRegistry  # noqa: E402

ROOT = SRC.parent
QUIT = {"q", "quit", "exit", "退出"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive Skills Harness CLI")
    parser.add_argument("-q", "--question", default=None, help="首轮问题")
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--once", action="store_true", help="只跑一轮")
    args = parser.parse_args()

    registry = SkillRegistry(ROOT / "skills")
    info = current_model_info()
    print("Progressive Skills Harness — CLI")
    print(f"模型: {info['display']}  （LLM_PROVIDER={info['provider']}）")
    print(f"已注册 Skills: {', '.join(s.name for s in registry.list_metas()) or '(无)'}")
    print("命令: quit 退出 | /skills 查看索引")
    print("-" * 50)

    pending = args.question
    while True:
        if pending is not None:
            question = pending.strip()
            pending = None
        else:
            try:
                question = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break

        if not question:
            continue
        if question.lower() in QUIT or question in QUIT:
            print("已退出。")
            break
        if question in ("/skills", "skills"):
            print(registry.build_index_text())
            continue

        run_and_print(question, max_steps=args.max_steps)
        if args.once:
            break


if __name__ == "__main__":
    main()
