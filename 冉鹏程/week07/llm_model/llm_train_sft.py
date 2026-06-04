import random
from pathlib import Path
import argparse
import torch
import json

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = '/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B'
OUTPUT_DIR = ROOT / "outputs"

# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（LoRA / 全量微调）")
    parser.add_argument("--model_path",  default=str(MODEL_PATH),
                        help="预训练模型的路径")
    parser.add_argument("--data_dir",    default=str(DATA_DIR),
                        help="数据目录路径")
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR),
                        help="输出目录路径")
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部 10748 条（默认）")
    parser.add_argument("--epochs",      default=3,    type=int,
                        help="训练轮数")
    parser.add_argument("--batch_size",  default=4,    type=int,
                        help="每个设备的批量大小")
    parser.add_argument("--grad_accum",  default=4,    type=int,
                        help="梯度累积步数，有效批量大小 = batch_size * grad_accum")
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=256,  type=int,
                        help="序列最大长度；NER 的 JSON 输出比分类长，建议 256")
    # 全量微调开关
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")
    # LoRA 超参（full_ft 时忽略）
    parser.add_argument("--lora_r",      default=8,    type=int,
                        help="LoRA 的秩，控制可训练参数数量")
    parser.add_argument("--lora_alpha",  default=16,   type=int,
                        help="LoRA 的缩放系数")
    parser.add_argument("--seed",        default=42,   type=int,
                        help="随机种子，保证实验可复现性")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备: cuda / cuda:0 / cuda:1 / cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    # 设置路径
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # 创建检查点目录

    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    # 读取训练集和验证集
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)
    


if __name__ == "__main__":
    main()