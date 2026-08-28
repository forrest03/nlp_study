"""GRPO 训练：Qwen3.5-0.8B 数学算术能力强化。

设计要点（简化自 grpo_arithmetic）：
  1. GRPO 无价值网络、无奖励模型：组内 K=8 条采样按奖励均值/标准差归一化出 advantage，
     beta=0 时不加载参考模型（省显存、省时间）。
  2. 复合奖励：正确分 1.0（宽松解析：<answer> 标签或最后一个数字）+ 格式分 0.2
     （输出带 <answer> 标签），两条曲线在日志中分别记录。
  3. 甜区选题：训练集只含三位数×两位数 / 三位数×三位数（基座正确率实测 ~70% / ~20%）；
     太易全对、太难全错的组 advantage 恒为 0，不产生梯度。
  4. 默认 LoRA：Qwen3.5-0.8B 全量微调的 AdamW 状态约 9GB+，8GB 显存必 OOM，
     默认 LoRA(r=16, lr=5e-5)。小学习率防止 RL 灾难性遗忘（实测 lr=2e-4 会
     把已会的简单题训崩：正确率 100%→20%）。
  5. 关闭 thinking 模式（enable_thinking=False）：Qwen3.5 默认带 <think> 标签，
     算术题直接输出答案，节省生成长度与训练时间。

用法：
  python main.py --train                          # 默认 LoRA 训练（100 步）
  python main.py --train --max_steps 3            # 冒烟测试：验证显存与流程
  python main.py --train --full                   # 全量微调（需 >12GB 显存）

输出：
  outputs/grpo_lora_ckpt/ 或 outputs/grpo_ckpt/   # checkpoint（含 tokenizer）
  outputs/train_log*.json                         # 每步指标
"""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 8GB 显存紧俏：扩展段分配减少碎片导致的 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from data import build_messages, make_problem, parse_output, pick_level

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = Path(r"F:\AI\programs\pretrain_models\Qwen3.5-0.8B")
OUT_DIR = ROOT / "outputs"

# Qwen3.5 混合注意力：4 层全注意力（q/k/v/o_proj）+ 20 层线性注意力
# （in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj），全部纳入 LoRA
# PEFT 按模块名子串匹配，无需写全路径
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
]


def build_dataset(n: int, seed: int) -> Dataset:
    """程序化生成训练集：prompt 为 chat 消息，answer 供奖励函数使用。"""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        level = pick_level(rng)
        expr, ans = make_problem(level, rng)
        rows.append({"prompt": build_messages(expr), "answer": ans})
    return Dataset.from_list(rows)


# ── 复合奖励：TRL 对多个 reward func 分别记录曲线，最后加权求和 ─────────────
def reward_correct(completions, answer, **kwargs):
    """正确分（宽松解析）：<answer> 标签内数字或输出最后一个数字正确即 1.0。"""
    rewards = []
    for comp, ans in zip(completions, answer):
        text = comp[0]["content"]
        rewards.append(1.0 if parse_output(text, int(ans))[2] else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    """格式分：输出包含 <answer>数字</answer> 即得分（与正确性解耦，0.2）。"""
    return [0.2 if parse_output(comp[0]["content"], 0)[0] else 0.0 for comp in completions]


def build_parser():
    parser = argparse.ArgumentParser(description="GRPO 训练 Qwen3.5-0.8B 数学能力")
    parser.add_argument("--max_steps", type=int, default=100, help="优化步数（每步 2 prompt × 8 采样）")
    parser.add_argument("--n_prompts", type=int, default=500, help="训练集 prompt 数")
    parser.add_argument("--full", action="store_true", help="全量微调（默认 LoRA，8GB 显存建议 LoRA）")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率（默认 LoRA 5e-5 / 全量 2e-6；过高会灾难性遗忘、训崩正确率）")
    parser.add_argument("--tag", type=str, default="", help="输出目录后缀，区分实验")
    parser.add_argument("--log_completions", action="store_true", help="打印每步真实采样补全（调试用）")
    return parser


def main(args=None):
    # 支持两种调用：python src/train.py 直接运行 / main.py 传入解析好的命名空间
    if args is None:
        args = build_parser().parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_dir = OUT_DIR / (f"grpo_ckpt{suffix}" if args.full else f"grpo_lora_ckpt{suffix}")
    log_path = OUT_DIR / (f"train_log{suffix}.json" if args.full else f"train_log_lora{suffix}.json")
    lr = args.lr if args.lr is not None else (2e-6 if args.full else 5e-5)

    dataset = build_dataset(args.n_prompts, seed=123)

    peft_config = None
    if not args.full:
        from peft import LoraConfig

        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS)

    config = GRPOConfig(
        output_dir=str(ckpt_dir),
        # 关键坑：config.json 虽声明 bfloat16，仍显式指定 dtype，避免 fp32 master
        # weights 下 AdamW eps 溢出导致 NaN（参考项目实测踩坑）
        model_init_kwargs={"dtype": "bfloat16"},
        # ── GRPO 核心参数 ─────────────────────────────────────────
        num_generations=8,          # 组内采样数 K：组内有对有错才有非零 advantage
        beta=0.0,                   # KL 系数 0：不加载参考模型，省显存
        epsilon=0.2,                # PPO-clip 裁剪范围
        temperature=1.0,            # 采样温度：保持组内多样性
        chat_template_kwargs={"enable_thinking": False},  # 关思考标签，直接出答案
        max_completion_length=128,  # 生成上限
        # ── 批次：4 completions/微批 × 累积 4 = 每步 2 prompt × 8 采样 ──
        # batch=8 时训练阶段 64 条 completion 的 logits 会把 8GB 显存打爆（实测 OOM），
        # 减半到 batch=4，累积步数翻倍保持每步训练量不变
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # 生成时 8×8=64 条并行会把显存推到 12GB+（溢出共享内存，速度暴跌），
        # 限制生成批次让峰值显存回落到 8GB 物理显存内
        generation_batch_size=32,
        # ── 训练超参 ──────────────────────────────────────────────
        learning_rate=lr,
        max_steps=args.max_steps,
        bf16=True,
        # 混合注意力 + gradient checkpointing 在 transformers 5.x 有兼容风险，关闭
        gradient_checkpointing=False,
        # ── 日志与保存 ────────────────────────────────────────────
        logging_steps=5,
        save_strategy="no",         # 只保存最终 checkpoint
        report_to=[],
        seed=42,
        log_completions=args.log_completions,
    )

    trainer = GRPOTrainer(
        model=str(BASE_MODEL),
        args=config,
        reward_funcs=[reward_correct, reward_format],
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    trainer.processing_class.save_pretrained(str(ckpt_dir))

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n训练完成。checkpoint: {ckpt_dir}")
    print(f"训练日志: {log_path}")
    print(f"GPU 峰值显存: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
