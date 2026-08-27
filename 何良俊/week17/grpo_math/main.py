"""统一入口：--train 训练 / --test 自动化测试。

用法：
  python main.py --train                          # GRPO 训练（默认 LoRA，100 步）
  python main.py --train --max_steps 3            # 冒烟测试训练流程
  python main.py --test                           # 自动化测试并保存 JSON
  python main.py --test --quick                   # 快速测试
  python main.py --test --model outputs/grpo_lora_ckpt   # 指定模型评估
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="基于 GRPO 的强化学习提升 Qwen3.5-0.8B 数学能力",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n  python main.py --train\n  python main.py --test\n  python main.py --test --quick",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="GRPO 训练模式")
    mode.add_argument("--test", action="store_true", help="自动化测试模式（结果保存为 JSON）")

    # ── 训练参数 ──────────────────────────────────────────────────────
    parser.add_argument("--max_steps", type=int, default=100, help="[train] 优化步数")
    parser.add_argument("--n_prompts", type=int, default=500, help="[train] 训练集 prompt 数")
    parser.add_argument("--full", action="store_true", help="[train] 全量微调（默认 LoRA）")
    parser.add_argument("--lr", type=float, default=None, help="[train] 学习率")
    parser.add_argument("--tag", type=str, default="", help="[train] 输出目录后缀")
    parser.add_argument("--log_completions", action="store_true", help="[train] 打印采样补全")

    # ── 测试参数 ──────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default=None, help="[test] 模型路径")
    parser.add_argument("--n", type=int, default=30, help="[test] 每难度题目数")
    parser.add_argument("--k", type=int, default=8, help="[test] pass@k 采样数")
    parser.add_argument("--seed", type=int, default=42, help="[test] 题目种子")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="[test] 生成长度上限")
    parser.add_argument("--quick", action="store_true", help="[test] 快速模式（每难度 5 题）")
    parser.add_argument("--out", type=str, default=None, help="[test] 结果 JSON 路径")
    args = parser.parse_args()

    if args.train:
        import train
        train.main(args)
    elif args.test:
        import test
        test.main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
