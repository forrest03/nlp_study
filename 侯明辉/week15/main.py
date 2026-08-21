"""
Week15 作业 CLI 入口。

用法:
  python main.py "你的问题"
  python main.py --corpus-dir /path/to/notes "你的问题"
  python main.py --demo   # 跑 3 个内置 demo 问题
"""
import argparse
import os
import sys
from pathlib import Path

# 让 python main.py 能找到 src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents import run_qa  # noqa: E402

DEMO_QUESTIONS = [
    "week15 的主题是什么？",
    "对比 week10、week11、week12 学了什么主题",
    "找所有提到 GraphRAG 的周次",
]


def main():
    parser = argparse.ArgumentParser(description="本地课程笔记多 subagent 问答")
    parser.add_argument("question", nargs="?", help="要问的问题")
    parser.add_argument("--corpus-dir", type=str, default=None,
                        help="文档目录（默认仓库根）")
    parser.add_argument("--demo", action="store_true",
                        help="跑 3 个内置 demo 问题")
    args = parser.parse_args()

    if not os.environ.get("SILICONFLOW_API_KEY"):
        parser.error("未配置 SILICONFLOW_API_KEY 环境变量，请参考 .env.example 设置后再运行")

    if args.demo and args.question:
        parser.error("--demo 与位置参数 question 互斥，请二选一")

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None

    if corpus_dir and not corpus_dir.is_dir():
        parser.error(f"corpus_dir 不存在或不是目录: {corpus_dir}")
    corpus_dir = corpus_dir.resolve() if corpus_dir else None

    if args.demo:
        for q in DEMO_QUESTIONS:
            print("\n" + "=" * 60)
            print(f"Demo 问题: {q}")
            print("=" * 60)
            r = run_qa(q, corpus_dir=corpus_dir, verbose=True)
            print(f"\n答案:\n{r['final_answer']}\n")
            print(f"（主 agent 用时 {r['duration']}s）")
    elif args.question:
        r = run_qa(args.question, corpus_dir=corpus_dir, verbose=True)
        print(f"\n答案:\n{r['final_answer']}\n")
        print(f"（主 agent 用时 {r['duration']}s）")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
