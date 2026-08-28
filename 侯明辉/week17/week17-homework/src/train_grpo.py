"""
GRPO 训练（扩展版）：Qwen2-0.5B-Instruct 扩展算术题（复合奖励）

相对 week17 主项目的差异：
  1. 训练集加入了 L8/L9/L10 三类扩展难度（应用题/括号/三数链）
  2. 训练集中保留 L3/L5 原主训难度（35%）—— 防止灾难性遗忘
  3. 答案统一用 float 存储（小数答案 L7 用得到，兼容整数）
  4. 配比原则：原能力 + 新能力，65% 新难度 + 35% 老难度

使用方式：
  python src/train_grpo.py                  # 完整训练（默认 200 步）
  python src/train_grpo.py --max_steps 3 --tag smoke   # 冒烟测试
  python src/train_grpo.py --lora           # 显存不足降级
  python src/train_grpo.py --include_l7     # 把 L7 小数也加入训练集
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset

import trl_compat  # noqa: F401  必须先于 trl 导入，修复 trl 0.21 + transformers 5.x 兼容
from trl import GRPOConfig, GRPOTrainer

from probe_baseline import SYSTEM_PROMPT, make_problem, parse_output

ROOT = Path(__file__).parent.parent
MODEL_PATH = Path(r"e:\File\Study\AI\AI-study\pretrain_models\Qwen2-0.5B-Instruct")
OUT_DIR = ROOT / "outputs"

# ── 训练集难度配比（基于 baseline_probe 50 题实测 informative_group_rate 调整）────
# 实测（每难度 50 题 × K=8）：
#   L3 0.82 / L5 0.76 / L7 0.64 / L2 0.68 —— GRPO 甜区（0.3~0.8），主训
#   L8 0.12 / L10 0.20 —— 保底探边界（极低 informative 但保留一些以防意外突破）
#   L1 0.58 / L4 0.70 / L6 0.28 / L9 0.18 —— 不训，留作泛化评估
#
# 设计原则：
#   1. 80% 集中在 L3/L5/L7（核心可学难度，主训）
#   2. 10% 保底 L2（防止两位数加减能力退化）
#   3. 10% 探边界 L8/L10（探一下能否突破能力下限）
#
# 注：L9 informative=0.18 但 50 题实测 pass@8=0.18（80 个采样只有 ~14 对），
# 训练组内方差太低，advantage 几乎为 0，纯浪费算力，所以不进训练集。
DEFAULT_LEVEL_MIX = [
    # 主训（informative 0.64~0.82，GRPO 甜区）
    ("L3_addsub_3digit",   0.35),
    ("L5_mul_2x1digit",    0.25),
    ("L7_decimal_addsub",  0.20),
    # 保底（informative 0.68，防止基础能力退化）
    ("L2_addsub_2digit",   0.10),
    # 探边界（informative < 0.3，看 RL 能否意外突破）
    ("L10_chain_3num",     0.05),
    ("L8_word_problem",    0.05),
]


def build_dataset(n: int, seed: int, level_mix) -> Dataset:
    """程序化生成训练集：prompt 为 chat 格式，answer/level 供 reward 函数使用。"""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r, acc, level = rng.random(), 0.0, level_mix[-1][0]
        for lv, p in level_mix:
            acc += p
            if r <= acc:
                level = lv
                break
        expr, ans = make_problem(level, rng)
        # 注意：ans 现在是 float（probe_baseline 扩展后统一为浮点数）
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"计算：{expr} = ?"},
                ],
                "answer": ans,            # float
                "level": level,
            }
        )
    return Dataset.from_list(rows)


# ── 复合奖励（与主项目一致，宽松解析应对冷启动）────────────────────────────────────
def reward_correct(completions, answer, **kwargs):
    """正确分（宽松解析）：有 <answer> 标签取标签内数字（含小数），否则取最后一个数字。

    注意：probe_baseline 把答案统一为 float，比较时也要转 float；
    parse_output 用 math.isclose 做浮点容差比较（abs_tol=1e-3）。
    """
    rewards = []
    for comp, ans in zip(completions, answer):
        text = comp[0]["content"]
        rewards.append(1.0 if parse_output(text, float(ans))[2] else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    """格式分：输出包含 <answer>数字</answer> 即得分（与正确性解耦）。"""
    return [0.2 if parse_output(comp[0]["content"], 0.0)[0] else 0.0 for comp in completions]


def parse_level_mix(include_l7: bool) -> list:
    """根据 --include_l7 决定是否把 L7 小数也加入训练集。"""
    mix = list(DEFAULT_LEVEL_MIX)
    if include_l7:
        # 把 L10 的一部分比例挪给 L7
        l10 = [lv for lv, p in mix if lv == "L10_chain_3num"][0]
        l10_p = [p for lv, p in mix if lv == "L10_chain_3num"][0]
        new_mix = [(lv, p) for lv, p in mix if lv != "L10_chain_3num"]
        new_mix.append(("L10_chain_3num", l10_p * 0.5))
        new_mix.append(("L7_decimal_addsub", l10_p * 0.5))
        return new_mix
    return mix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200, help="优化步数（每步 4 prompt × 8 采样）")
    parser.add_argument("--n_prompts", type=int, default=1000, help="训练集 prompt 数")
    parser.add_argument("--lr", type=float, default=2e-6, help="全量微调学习率")
    parser.add_argument("--lora", action="store_true", help="降级为 LoRA（全量 OOM 时使用）")
    parser.add_argument("--tag", type=str, default="", help="输出目录后缀")
    parser.add_argument("--log_completions", action="store_true", help="打印每步真实采样补全")
    parser.add_argument("--include_l7", action="store_true",
                        help="把 L7 小数加减加入训练集（默认不训，留作泛化）")
    args = parser.parse_args()

    level_mix = parse_level_mix(args.include_l7)
    print(f"[训练集难度配比]")
    for lv, p in level_mix:
        print(f"  {lv:<22} {p:.0%}")
    print()

    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_dir = OUT_DIR / (f"grpo_lora_ckpt{suffix}" if args.lora else f"grpo_ckpt{suffix}")
    log_path = OUT_DIR / (f"train_log_lora{suffix}.json" if args.lora else f"train_log{suffix}.json")

    dataset = build_dataset(args.n_prompts, seed=123, level_mix=level_mix)

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    config = GRPOConfig(
        output_dir=str(ckpt_dir),
        # 本地 Qwen2-0.5B-Instruct 的 config.json 写的是 torch_dtype=bfloat16，
        # 默认会按 bf16 加载 —— 无需显式指定。但 trl 1.12 的新 API 推荐用 `dtype`。
        # 保留 model_init_kwargs 作为显式声明，便于排查 dtype 问题。
        model_init_kwargs={"dtype": "bfloat16"},
        # ── GRPO 核心参数 ─────────────────────────────────────────────
        num_generations=8,
        beta=0.0,
        epsilon=0.2,
        temperature=1.0,
        # max_prompt_length 在 trl 1.12 移除：prompt 长度由 tokenize 阶段控制
        max_completion_length=128,              # 扩展题答案也较长
        # ── 批次 ──────────────────────────────────────────────────────
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        # ── 训练超参 ──────────────────────────────────────────────────
        learning_rate=args.lr if not args.lora else 2e-4,
        max_steps=args.max_steps,
        bf16=True,
        # 关键：关闭 gradient_checkpointing（0.5B 模型显存够用，开启会拖慢训练）
        gradient_checkpointing=False,
        # ── 日志与保存 ────────────────────────────────────────────────
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=42,
        log_completions=args.log_completions,
    )

    trainer = GRPOTrainer(
        model=str(MODEL_PATH),
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