"""
基线摸底（扩展版）：Qwen2-0.5B-Instruct 在扩展算术题难度上的表现

作业扩展（相对于 week17 主项目 L1~L6）：
  L7  小数加减（保留 1~2 位小数，如 3.5 + 2.7 = 6.2）
  L8  两步应用题（如"30 元买 3 个 5 元/个 + 1 个 7 元，找回 8 元"）
  L9  带括号混合运算（如 (5 + 3) × 2 - 4 = 12）
  L10 三数运算链（a + b + c 或 a × b + c 的混合）

核心改动：
  1. make_problem() 增加 L7~L10 四个级别
  2. parse_output() 支持小数（浮点数比较，处理 6.0 vs 6）
  3. NUM_RE / TAG_RE 扩展为支持小数
  4. 训练集难度配比基于新的 informative rate 调整

使用方式：
  python src/probe_baseline.py             # 全量摸底（10 难度 × 50 题，K=8）
  python src/probe_baseline.py --quick     # 快速验证（每难度 10 题）
  python src/probe_baseline.py --model outputs/grpo_ckpt --out outputs/post_train_probe.json --seed 42
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
MODEL_PATH = Path(r"e:\File\Study\AI\AI-study\pretrain_models\Qwen2-0.5B-Instruct")
OUT_PATH = ROOT / "outputs" / "baseline_probe.json"

SYSTEM_PROMPT = (
    "你是一个算术助手。用户会给你一道算术题，请计算出结果，"
    "并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。"
    "不要输出其他内容。"
)

# ── 扩展：支持整数和小数（带可选小数部分）────────────────────────────────────
TAG_RE = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# 注：-?\d+(?:\.\d+)? 可以匹配 "6", "6.0", "6.00", "-3.14" 等


def make_problem(level: str, rng: random.Random):
    """按难度级别生成一道算术题，返回 (表达式文本, 标准答案)。

    答案统一用浮点数表示（即使是整数也用 float），便于小数题目统一比较。
    """
    if level == "L1_add_1digit":          # 个位数加法
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return f"{a} + {b}", float(a + b)
    if level == "L2_addsub_2digit":       # 两位数加减
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            return f"{a} + {b}", float(a + b)
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", float(a - b)
    if level == "L3_addsub_3digit":       # 三位数加减
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if rng.random() < 0.5:
            return f"{a} + {b}", float(a + b)
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", float(a - b)
    if level == "L4_mul_1digit":          # 表内乘法
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        return f"{a} × {b}", float(a * b)
    if level == "L5_mul_2x1digit":        # 两位数乘一位数
        a, b = rng.randint(10, 99), rng.randint(3, 9)
        return f"{a} × {b}", float(a * b)
    if level == "L6_mul_2x2digit":        # 两位数乘两位数
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"{a} × {b}", float(a * b)

    # ── 扩展难度 L7~L10 ─────────────────────────────────────────────────────
    if level == "L7_decimal_addsub":      # 小数加减（1~2 位小数）
        # 生成小数：用整数 + 随机小数部分
        a_int = rng.randint(10, 99)
        b_int = rng.randint(10, 99)
        a_dec = rng.choice([0, 0.5, 0.25, 0.75, 0.1, 0.2, 0.8, 0.9])
        b_dec = rng.choice([0, 0.5, 0.25, 0.75, 0.1, 0.2, 0.8, 0.9])
        a = round(a_int + a_dec, 2)
        b = round(b_int + b_dec, 2)
        if rng.random() < 0.5:
            return f"{a} + {b}", round(a + b, 2)
        a, b = max(a, b), min(a, b)
        return f"{a} - {b}", round(a - b, 2)

    if level == "L8_word_problem":        # 两步应用题
        # 模板：小明有 X 元，买了 A 个 B 元/个 的 X1，又买了 C 个 D 元/个 的 X2，
        #       还剩多少元？  （要求 X - A*B - C*D 仍 >= 0）
        money = rng.randint(50, 99)
        n1 = rng.randint(2, 4)
        p1 = rng.randint(3, 9)
        cost1 = n1 * p1
        n2 = rng.randint(1, 3)
        p2 = rng.randint(3, 9)
        cost2 = n2 * p2
        # 保证 money >= cost1 + cost2
        while money < cost1 + cost2:
            money = rng.randint(50, 99)
            cost1 = n1 * p1
            cost2 = n2 * p2
        remain = money - cost1 - cost2
        text = (
            f"小明有 {money} 元，买了 {n1} 个单价 {p1} 元的苹果"
            f"（共 {cost1} 元），又买了 {n2} 个单价 {p2} 元的橘子"
            f"（共 {cost2} 元），还剩多少元？"
        )
        return text, float(remain)

    if level == "L9_paren_mixed":         # 带括号混合运算
        # 模板：(a + b) × c - d 或 (a - b) × c + d
        a = rng.randint(10, 50)
        b = rng.randint(1, 9)
        c = rng.randint(2, 9)
        d = rng.randint(1, 20)
        if rng.random() < 0.5:
            expr = f"({a} + {b}) × {c} - {d}"
            ans = (a + b) * c - d
        else:
            a, b = max(a, b), min(a, b)
            expr = f"({a} - {b}) × {c} + {d}"
            ans = (a - b) * c + d
        return expr, float(ans)

    if level == "L10_chain_3num":         # 三数运算链
        # 模板：a + b + c、a + b × c、a × b + c
        a = rng.randint(5, 50)
        b = rng.randint(2, 20)
        c = rng.randint(2, 9)
        choice = rng.choice(["sum", "mixed1", "mixed2"])
        if choice == "sum":                # a + b + c（纯加法）
            expr = f"{a} + {b} + {c}"
            ans = a + b + c
        elif choice == "mixed1":           # a + b × c（乘法优先）
            expr = f"{a} + {b} × {c}"
            ans = a + b * c
        else:                              # a × b + c（乘法优先）
            expr = f"{a} × {b} + {c}"
            ans = a * b + c
        return expr, float(ans)

    raise ValueError(level)


LEVELS = [
    "L1_add_1digit",
    "L2_addsub_2digit",
    "L3_addsub_3digit",
    "L4_mul_1digit",
    "L5_mul_2x1digit",
    "L6_mul_2x2digit",
    # ── 扩展难度 ─────────────────────────────────────────────────────────
    "L7_decimal_addsub",
    "L8_word_problem",
    "L9_paren_mixed",
    "L10_chain_3num",
]


def parse_output(text: str, answer: float, tol: float = 1e-3):
    """解析模型输出，返回 (是否符合格式, 严格正确, 宽松正确)。

    严格：必须有 <answer>X</answer> 且 X 与 answer 浮点相等（误差 tol）
    宽松：取输出中最后一个数字（含小数）与 answer 浮点相等
    """
    m = TAG_RE.search(text)
    fmt_ok = m is not None
    if fmt_ok:
        try:
            strict_ok = math.isclose(float(m.group(1)), answer, abs_tol=tol, rel_tol=0)
        except ValueError:
            strict_ok = False
    else:
        strict_ok = False
    nums = NUM_RE.findall(text)
    loose_ok = False
    if nums:
        try:
            loose_ok = math.isclose(float(nums[-1]), answer, abs_tol=tol, rel_tol=0)
        except ValueError:
            loose_ok = False
    return fmt_ok, strict_ok, loose_ok


def build_prompts(tokenizer, problems):
    texts = []
    for expr, _ in problems:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"计算：{expr} = ?"},
        ]
        texts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return texts


@torch.no_grad()
def generate(model, tokenizer, texts, do_sample, k=1, batch_size=16, max_new_tokens=128):
    """分批生成。do_sample=True 时每条 prompt 返回 k 个样本，外层列表按 prompt 对齐。

    注意：扩展题（如两步应用题）答案较长，max_new_tokens 提到 128。
    """
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=1.0 if do_sample else None,
            top_p=1.0 if do_sample else None,
            num_return_sequences=k if do_sample else 1,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        if do_sample:
            all_outputs.extend(
                decoded[j * k : (j + 1) * k] for j in range(len(batch))
            )
        else:
            all_outputs.extend(decoded)
    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="每难度只跑 10 题，快速验证")
    parser.add_argument("--n", type=int, default=50, help="每个难度级别的题目数")
    parser.add_argument("--k", type=int, default=8, help="pass@k 的采样数")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--out", type=str, default=str(OUT_PATH))
    parser.add_argument("--seed", type=int, default=42, help="题目生成随机种子")
    args = parser.parse_args()
    n = 10 if args.quick else args.n

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (Path(args.model) / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="cuda"
        )
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
    model.eval()

    rng = random.Random(args.seed)
    report = {}

    for level in LEVELS:
        t0 = time.time()
        problems = [make_problem(level, rng) for _ in range(n)]
        texts = build_prompts(tokenizer, problems)

        # ── 1. greedy 单样本 ─────────────────────────────────────────
        greedy_outs = generate(model, tokenizer, texts, do_sample=False)
        greedy_fmt = greedy_strict = greedy_loose = 0
        for (expr, ans), out in zip(problems, greedy_outs):
            fmt, strict, loose = parse_output(out, ans)
            greedy_fmt += fmt
            greedy_strict += strict
            greedy_loose += loose

        # ── 2. 温度采样 k 条 ──────────────────────────────────────────
        sample_outs = generate(model, tokenizer, texts, do_sample=True, k=args.k)
        sample_strict_sum = sample_loose_sum = 0
        pass_at_k = loose_pass_at_k = 0
        mixed_groups = loose_mixed_groups = 0
        for (_, ans), outs in zip(problems, sample_outs):
            results = [parse_output(o, ans) for o in outs]
            n_strict = sum(r[1] for r in results)
            n_loose = sum(r[2] for r in results)
            sample_strict_sum += n_strict
            sample_loose_sum += n_loose
            pass_at_k += n_strict > 0
            loose_pass_at_k += n_loose > 0
            mixed_groups += 0 < n_strict < args.k
            loose_mixed_groups += 0 < n_loose < args.k

        report[level] = {
            "n": n,
            "k": args.k,
            "greedy_format_rate": round(greedy_fmt / n, 4),
            "greedy_strict_acc": round(greedy_strict / n, 4),
            "greedy_loose_acc": round(greedy_loose / n, 4),
            "sample_strict_acc": round(sample_strict_sum / (n * args.k), 4),
            "sample_loose_acc": round(sample_loose_sum / (n * args.k), 4),
            f"pass@{args.k}": round(pass_at_k / n, 4),
            f"loose_pass@{args.k}": round(loose_pass_at_k / n, 4),
            "informative_group_rate": round(mixed_groups / n, 4),
            "loose_informative_group_rate": round(loose_mixed_groups / n, 4),
            "elapsed_sec": round(time.time() - t0, 1),
            "examples": [
                {"expr": expr, "answer": ans, "greedy_output": out}
                for (expr, ans), out in list(zip(problems, greedy_outs))[:3]
            ],
        }
        r = report[level]
        print(
            f"{level:<22} greedy_loose={r['greedy_loose_acc']:.2f} "
            f"fmt={r['greedy_format_rate']:.2f} "
            f"loose_acc={r['sample_loose_acc']:.2f} "
            f"loose_pass@{args.k}={r[f'loose_pass@{args.k}']:.2f} "
            f"loose_informative={r['loose_informative_group_rate']:.2f} "
            f"({r['elapsed_sec']}s)"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n结果已保存：{out_path}")
    print(f"GPU 峰值显存：{peak_gb:.2f} GB")


if __name__ == "__main__":
    main()