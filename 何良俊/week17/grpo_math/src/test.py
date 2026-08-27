"""自动化测试：评估模型在各难度算术题上的表现，结果保存为 JSON。

指标说明：
  greedy_*            确定性解码（temperature=0）下的格式遵循率 / 严格正确率 / 宽松正确率
  sample_*            温度采样 K 条后逐条统计的宽松正确率
  pass@k              每组 K 条中至少一条宽松正确的比例
  informative_group_rate  0<正确数<k 的组占比 —— GRPO 能学到东西的训练燃料比例

用法：
  python main.py --test                            # 自动选择最新 checkpoint（无则测基座）
  python main.py --test --model outputs/grpo_lora_ckpt
  python main.py --test --n 10 --quick             # 快速验证

输出：
  outputs/test_result.json                          # 全量结果（含样例输出）
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import LEVELS, build_messages, make_problem, parse_output

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = Path(r"F:\AI\programs\pretrain_models\Qwen3.5-0.8B")
OUT_DIR = ROOT / "outputs"
DEFAULT_OUT = OUT_DIR / "test_result.json"


def resolve_model(path: str | None) -> str:
    """默认模型选择：--model > 最新 checkpoint > 基座模型。"""
    if path:
        return path
    for ckpt in ["grpo_lora_ckpt", "grpo_ckpt"]:
        p = OUT_DIR / ckpt
        if p.exists():
            return str(p)
    return str(BASE_MODEL)


def load_model(path: str):
    """加载模型与 tokenizer；LoRA checkpoint 自动挂载到基座。

    关键坑：Qwen3.5-0.8B 是多模态模型（architectures=Qwen3_5ForConditionalGeneration），
    TRL 训练时把 LoRA 注入到文本子模块 model.language_model，保存的 adapter key 带
    language_model 前缀。测试端必须用 AutoModelForImageTextToText 加载（而不是
    AutoModelForCausalLM），否则 adapter 权重路径对不上、被随机初始化。
    """
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if (Path(path) / "adapter_config.json").exists():
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText

        base = AutoModelForImageTextToText.from_pretrained(
            str(BASE_MODEL), dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, texts, do_sample, k=1, batch_size=16, max_new_tokens=128):
    """分批生成。do_sample=True 时每条 prompt 返回 k 个样本，外层列表按 prompt 对齐。"""
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
        if do_sample:  # num_return_sequences 把每条 prompt 的 k 个样本连续排列
            all_outputs.extend(decoded[j * k : (j + 1) * k] for j in range(len(batch)))
        else:
            all_outputs.extend(decoded)
    return all_outputs


def evaluate(model, tokenizer, level, n, k, seed, max_new_tokens):
    rng = random.Random(seed)
    problems = [make_problem(level, rng) for _ in range(n)]
    texts = [
        tokenizer.apply_chat_template(
            build_messages(expr), tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        for expr, _ in problems
    ]

    # ── 1. greedy 单样本：确定性能力 + 格式遵循 ──────────────────────────
    greedy_outs = generate(model, tokenizer, texts, do_sample=False,
                           max_new_tokens=max_new_tokens)
    greedy_fmt = greedy_strict = greedy_loose = 0
    for (expr, ans), out in zip(problems, greedy_outs):
        fmt, strict, loose = parse_output(out, ans)
        greedy_fmt += fmt
        greedy_strict += strict
        greedy_loose += loose

    # ── 2. 温度采样 k 条：pass@k 与 informative group rate ──────────────
    sample_outs = generate(model, tokenizer, texts, do_sample=True, k=k,
                           max_new_tokens=max_new_tokens)
    sample_loose_sum = 0
    loose_pass_at_k = 0
    loose_mixed_groups = 0  # 0 < 正确数 < k：GRPO 真正能学到东西的组
    for (_, ans), outs in zip(problems, sample_outs):
        n_loose = sum(parse_output(o, ans)[2] for o in outs)
        sample_loose_sum += n_loose
        loose_pass_at_k += n_loose > 0
        loose_mixed_groups += 0 < n_loose < k

    report = {
        "n": n,
        "k": k,
        "greedy_format_rate": round(greedy_fmt / n, 4),
        "greedy_strict_acc": round(greedy_strict / n, 4),
        "greedy_loose_acc": round(greedy_loose / n, 4),
        "sample_loose_acc": round(sample_loose_sum / (n * k), 4),
        f"pass@{k}": round(loose_pass_at_k / n, 4),
        "informative_group_rate": round(loose_mixed_groups / n, 4),
        "examples": [
            {"expr": expr, "answer": ans, "greedy_output": out}
            for (expr, ans), out in list(zip(problems, greedy_outs))[:3]
        ],
    }
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="自动化测试模型数学能力")
    parser.add_argument("--model", type=str, default=None, help="模型路径（默认选最新 checkpoint，无则基座）")
    parser.add_argument("--n", type=int, default=30, help="每个难度级别的题目数")
    parser.add_argument("--k", type=int, default=8, help="pass@k 采样数（与 GRPO 组大小一致）")
    parser.add_argument("--seed", type=int, default=42, help="题目种子（换种子避免与训练集重叠）")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="生成长度上限")
    parser.add_argument("--quick", action="store_true", help="快速模式：每难度 5 题")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="结果 JSON 输出路径")
    return parser


def main(args=None):
    # 支持两种调用：python src/test.py 直接运行 / main.py 传入解析好的命名空间
    if args is None:
        args = build_parser().parse_args()
    else:
        args.out = args.out or str(DEFAULT_OUT)

    n = 5 if args.quick else args.n
    model_path = resolve_model(args.model)
    print(f"评估模型: {model_path}")

    model, tokenizer = load_model(model_path)
    report = {"meta": {
        "model": model_path,
        "base_model": str(BASE_MODEL),
        "seed": args.seed,
        "n_per_level": n,
        "k": args.k,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, "per_level": {}}

    t0 = time.time()
    for level in LEVELS:
        r = evaluate(model, tokenizer, level, n, args.k, args.seed, args.max_new_tokens)
        report["per_level"][level] = r
        print(
            f"{level:<20} greedy_loose={r['greedy_loose_acc']:.2f} "
            f"format={r['greedy_format_rate']:.2f} "
            f"sample_loose={r['sample_loose_acc']:.2f} "
            f"pass@{args.k}={r[f'pass@{args.k}']:.2f} "
            f"informative={r['informative_group_rate']:.2f}"
        )

    # ── 汇总：各难度简单平均 ─────────────────────────────────────────────
    keys = ["greedy_format_rate", "greedy_strict_acc", "greedy_loose_acc",
            "sample_loose_acc", f"pass@{args.k}", "informative_group_rate"]
    report["summary"] = {
        k: round(sum(report["per_level"][lv][k] for lv in LEVELS) / len(LEVELS), 4)
        for k in keys
    }
    s = report["summary"]
    print(
        f"\n{'平均':<20} greedy_loose={s['greedy_loose_acc']:.2f} "
        f"format={s['greedy_format_rate']:.2f} "
        f"sample_loose={s['sample_loose_acc']:.2f} "
        f"pass@{args.k}={s[f'pass@{args.k}']:.2f} "
        f"informative={s['informative_group_rate']:.2f}"
    )
    print(f"总耗时: {time.time() - t0:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存：{out_path}")


if __name__ == "__main__":
    main()
