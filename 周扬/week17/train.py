# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════
 GRPO 强化学习 —— 基于 traindata.json 的完整实现
════════════════════════════════════════════════════════════════════════

任务：用 GRPO（Group Relative Policy Optimization）在 traindata.json
      （1~4 位数加减乘除，共 1600 条）上训练小模型，学会
      "输出 <ans>N</ans> 格式" + "算得对"。

参考教学示例：grpo_arithmetic/src/train_grpo.py（只读）

环境说明（本机实测）：
  - conda py312 环境，trl 1.10.0 / transformers 5.9 / torch 2.12
  - 无 CUDA，只能用 MPS 或 CPU → 训练用 fp32（bf16 仅 CUDA 下启用）
  - trl 1.10 原生兼容 transformers 5.x，不需要 trl_compat 补丁
    （那个补丁专修 trl 0.21，与本环境无关）

运行：
  python train.py                  # 完整训练（默认 max_steps=40）
  python train.py --max_steps 3    # 冒烟测试
════════════════════════════════════════════════════════════════════════
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer

# ── 路径与常量 ────────────────────────────────────────────────────────
MODEL_PATH = "/Users/zhouyang/models/Qwen2.5-0.5B-Instruct"   # 本机模型
TRAIN_DATA_PATH = "traindata.json"                            # 你的训练数据
OUT_DIR = "outputs"

# 系统提示词：指令模型需要先了解任务 + 输出格式。标签用 <ans>，
# 与下方 parse_output()/reward 的格式判定必须一致。
SYSTEM_PROMPT = (
    "你是一个算术助手。用户会给你一道算术题，请计算出结果，"
    "并把最终答案放在 <ans> 标签中，例如 <ans>42</ans>。"
    "不要输出其他内容。"
)

# ═══════════════════════════ 第 1 步 · 数据加载 ═══════════════
# traindata.json 每条：{ "位数","type","prompt","expression","answer" }，
# answer 是字符串。这里转成 trl 需要的 Dataset：prompt 用 chat 消息列表。
def load_data(n=None, seed=42):
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None:
        rows = rows[:n]
    ds_rows = []
    for r in rows:
        ds_rows.append({
            # trl 会对 "prompt" 列自动应用 chat 模板
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": r["prompt"]},
            ],
            "answer": int(r["answer"]),
            # 难度标签：用于观察不同难度对奖励收敛的影响
            "level": f'{r["位数"]}_{r["type"]}',
        })
    return Dataset.from_list(ds_rows)


# ═══════════════════════════ 第 2 步 · 输出解析 ═══════════════
TAG_RE = re.compile(r"<ans>\s*(-?\d+)\s*</ans>")   # 严格格式标签
NUM_RE = re.compile(r"-?\d+")                        # 宽松兜底：取最后一个整数


def parse_output(text, answer):
    """返回 (格式是否OK, 严格对, 宽松对)。"""
    m = TAG_RE.search(text)
    fmt_ok = m is not None
    strict_ok = fmt_ok and int(m.group(1)) == answer
    nums = NUM_RE.findall(text)
    loose_ok = bool(nums) and int(nums[-1]) == answer
    return fmt_ok, strict_ok, loose_ok


# ═══════════════════════════ 第 3 步 · 奖励函数 ═══════════════
# 两个可叠加的 reward：正确分 1.0 + 格式分 0.2。TRL 分别记录曲线，训练时求和。
def reward_correct(completions, answer, **kwargs):
    """正确分（宽松解析）：训练早期模型不出 <ans> 标签，用宽松口径保冷启动有梯度。"""
    rewards = []
    for comp, ans in zip(completions, answer):
        text = comp[0]["content"]
        rewards.append(1.0 if parse_output(text, int(ans))[2] else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    """格式分 0.2：输出含 <ans>数字</ans> 即得分（与正确性解耦）。"""
    return [0.2 if parse_output(comp[0]["content"], 0)[0] else 0.0
            for comp in completions]


# ═══════════════════════════ 第 4 步 · 训练主流程 ═══════════════
def train_main(max_steps=40, n_prompts=None, lr=2e-6, tag="", seed=42):
    suffix = f"_{tag}" if tag else ""
    ckpt_dir = Path(OUT_DIR) / f"grpo_ckpt{suffix}"
    log_path = Path(OUT_DIR) / f"train_log{suffix}.json"

    # 设备感知：本机无 CUDA，用 MPS/CPU → fp32；CUDA → bf16（防止 fp16 的 AdamW 溢出）
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    use_bf16 = device == "cuda"
    print(f"[device] {device}  (bf16={use_bf16})")

    dataset = load_data(n_prompts, seed=seed)
    print(f"[data]  训练集 {len(dataset)} 条，分布：")
    from collections import Counter
    for k, v in sorted(Counter(dataset["level"]).items()):
        print(f"        {k}: {v}")

    config = GRPOConfig(
        output_dir=str(ckpt_dir),
        model_init_kwargs={"torch_dtype": "bfloat16" if use_bf16 else "float32"},
        # ── GRPO 核心参数 ──────────────────────────────────
        num_generations=8,      # 组内采样数 K：group-based advantage 估计
        beta=0.0,               # KL 系数=0：不加载参考模型，省显存/内存
        epsilon=0.2,            # PPO-clip 裁剪范围
        temperature=1.0,        # 采样温度：保证组内多样性（GRPO 的训练燃料）
        max_completion_length=64,
        # ── 批次：8 completions/微批 × 累积 4 = 每步 4 prompt × 8 采样 ──
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        # ── 训练超参 ────────────────────────────────────────
        learning_rate=lr,
        max_steps=max_steps,
        bf16=use_bf16,
        gradient_checkpointing=False,  # 若开了会损坏 generate 采样，务必保持关
        # ── 日志与保存 ─────────────────────────────────────
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=seed,
    )

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        args=config,
        reward_funcs=[reward_correct, reward_format],
        train_dataset=dataset,
    )
    trainer.train()  # GRPO 在线采样训练

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    trainer.processing_class.save_pretrained(str(ckpt_dir))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"\n[训练完成] checkpoint: {ckpt_dir}")
    print(f"[训练日志] {log_path}")

    # 打印每步指标（reward 分量、熵、clip 比例等）里的最后一条
    if trainer.state.log_history:
        for h in reversed(trainer.state.log_history):
            if "rewards/reward_correct" in h and "loss" in h:
                print("[末步指标]", {k: h[k] for k in h if not k.startswith("grad")})
                break


# ═══════════════════════════ 第 5 步 · 训练前后评估 ═══════════════
@torch.no_grad()
def generate(model, tokenizer, texts, do_sample, k=1, batch_size=16, max_new_tokens=64):
    """分批生成；do_sample=True 时每条 prompt 返回 k 个样本，外层按 prompt 对齐。"""
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
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
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        if do_sample:
            all_outputs.extend(decoded[j * k:(j + 1) * k] for j in range(len(batch)))
        else:
            all_outputs.extend(decoded)
    return all_outputs


def device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(ckpt_path):
    """加载模型与 tokenizer；CUDA→bf16，MPS/CPU→auto(fp32)。设备映射交给 backend。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        dtype=torch.bfloat16 if device_name() == "cuda" else "auto")
    model.to(device_name())
    model.eval()
    return model, tokenizer


def build_questions(n=50, seed=777):
    """抽一批问题（固定 seed，训练前后用同一批）。返回 (prompt, answer, level) 列表。"""
    flat = json.load(open(TRAIN_DATA_PATH, encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(flat)
    return [(r["prompt"], int(r["answer"]), f'{r["位数"]}_{r["type"]}') for r in flat[:n]]


def run_eval(model, tokenizer, problems, k=8, max_new_tokens=64):
    """对同一批问题做 greedy + 温度采样，逐题收集输出与判定。"""
    msgs = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": p}] for p, _, _ in problems
    ]
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in msgs]

    greedy_outs = generate(model, tokenizer, texts, do_sample=False, max_new_tokens=max_new_tokens)
    sample_outs = generate(model, tokenizer, texts, do_sample=True, k=k, max_new_tokens=max_new_tokens)

    records = []
    for i, (p, ans, level) in enumerate(problems):
        gf, gs, gl = parse_output(greedy_outs[i], ans)
        samples = [parse_output(o, ans) for o in sample_outs[i]]
        n_loose = sum(r[2] for r in samples)
        records.append({
            "prompt": p,
            "answer": str(ans),
            "level": level,
            "greedy_output": greedy_outs[i],
            "greedy_format": gf,
            "greedy_correct": gl,
            "sample_outputs": sample_outs[i],
            "sample_n_correct": n_loose,
            "sample_n_total": k,
        })
    return records


def summarize(records, k=8):
    """从逐题记录算聚合指标。"""
    n = len(records)
    return {
        "n": n,
        "greedy_format_rate": round(sum(r["greedy_format"] for r in records) / n, 4),
        "greedy_acc": round(sum(r["greedy_correct"] for r in records) / n, 4),
        f"pass@{k}": round(sum(r["sample_n_correct"] > 0 for r in records) / n, 4),
        "informative_group_rate": round(
            sum(0 < r["sample_n_correct"] < k for r in records) / n, 4),
    }


def evaluate(ckpt_path, out=None, n=50, k=8, seed=777):
    """评估某 checkpoint，返回 (records, metrics)，可选写结果 json。"""
    model, tokenizer = load_model_and_tokenizer(ckpt_path)
    problems = build_questions(n, seed)
    records = run_eval(model, tokenizer, problems, k=k)
    metrics = summarize(records, k=k)
    print(f"[eval] {ckpt_path}  格式率={metrics['greedy_format_rate']} "
          f"正确率={metrics['greedy_acc']} pass@{k}={metrics[f'pass@{k}']}")
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"metrics": metrics, "records": records}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[eval] 逐题结果已保存: {out}")
    return records, metrics


def metric_by_level(records):
    """按难度级别分组统计。"""
    groups = {}
    for r in records:
        groups.setdefault(r["level"], []).append(r)
    out = {}
    for lv, recs in groups.items():
        out[lv] = summarize(recs, k=8 if recs else 0)
    return out


# ═══════════════════════════ 第 6 步 · 对比分析报告 ═══════════════
def write_report(base_rec, post_rec, report_path="outputs/compare_report.md"):
    """把训练前/后两份逐题结果写成人话对比报告。"""
    bm, pm = base_rec["metrics"], post_rec["metrics"]
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    L = []
    L.append("# GRPO 强化学习前后对比报告\n")
    L.append(f"- 数据：`traindata.json`（1~4 位加减乘除，共 1600 条）")
    L.append(f"- 模型：`{MODEL_PATH}`")
    L.append(f"- 评估题数：{bm['n']}（固定 seed，训练前后为同一批问题）\n")

    L.append("## 一、整体指标\n")
    metric_names = ["greedy_format_rate", "greedy_acc", "pass@8", "informative_group_rate"]
    labels = {"greedy_format_rate": "格式遵循率", "greedy_acc": "greedy 正确率",
              "pass@8": "pass@8", "informative_group_rate": "informative组比例"}
    L.append("| 指标 | 训练前 | 训练后 | 提升 |")
    L.append("|---|---:|---:|---:|")
    for m in metric_names:
        b, p = bm[m], pm[m]
        delta = p - b
        L.append(f"| {labels[m]} | {b:.4f} | {p:.4f} | {delta:+.4f} |")
    L.append("")

    L.append("## 二、按难度分组\n")
    for level in sorted(set(list(base_rec["metric_by_level"]) + list(post_rec["metric_by_level"]))):
        b_ = base_rec["metric_by_level"].get(level, {"n": 0, "greedy_format_rate": 0, "greedy_acc": 0, "pass@8": 0})
        p_ = post_rec["metric_by_level"].get(level, {"n": 0, "greedy_format_rate": 0, "greedy_acc": 0, "pass@8": 0})
        L.append(f"### {level}\n")
        L.append("| 指标 | 训练前 | 训练后 |")
        L.append("|---|---:|---:|")
        L.append(f"| 格式遵循率 | {b_['greedy_format_rate']:.4f} | {p_['greedy_format_rate']:.4f} |")
        L.append(f"| greedy 正确率 | {b_['greedy_acc']:.4f} | {p_['greedy_acc']:.4f} |")
        L.append(f"| pass@8 | {b_.get('pass@8', 0):.4f} | {p_.get('pass@8', 0):.4f} |")
        L.append("")

    L.append("## 三、逐题对照（节选）\n")
    shown = 0
    for br, pr in zip(base_rec["records"], post_rec["records"]):
        if shown >= 20:
            break
        marker = ""
        if br["greedy_correct"] and not pr["greedy_correct"]:
            marker = "（回归）"
        elif not br["greedy_correct"] and pr["greedy_correct"]:
            marker = "（改善）"
        L.append(f"1. **{br['prompt']}** → 答案 `{br['answer']}` {marker}")
        L.append(f"   - 前: `{br['greedy_output'].strip()}`  {'✓' if br['greedy_correct'] else '✗'}")
        L.append(f"   - 后: `{pr['greedy_output'].strip()}`  {'✓' if pr['greedy_correct'] else '✗'}")
        shown += 1
    L.append("")

    L.append("## 四、结论\n")
    L.append(f"- 格式遵循率：{bm['greedy_format_rate']:.2f} → {pm['greedy_format_rate']:.2f}"
             f"（{'提升' if pm['greedy_format_rate'] > bm['greedy_format_rate'] else '持平/下降'}）")
    L.append(f"- greedy 正确率：{bm['greedy_acc']:.2f} → {pm['greedy_acc']:.2f}"
             f"（{'明显爬升，RL 生效' if pm['greedy_acc'] - bm['greedy_acc'] > 0.05 else '变化不大'}）")
    L.append("- 观察点：4 位×4 位等超出 0.5B 能力边界的题，组内常全错 → advantage 全 0 → "
             "RL 无法凭空创造能力，正确率提升有限，这是 GRPO 的教科书结论。")

    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] 对比报告已生成: {report_path}")


# ═══════════════════════════ 全流程 · 训练前→训练→训练后→报告 ═══
def pipeline(max_steps=40, n_eval=50, k=8, lr=2e-6, eval_seed=777, tag=""):
    # 1) 训练前：用基座模型回答同一批问题
    print("\n════ 阶段 1/4 · 训练前基线评估 ════")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    problems = build_questions(n_eval, eval_seed)
    base_records = run_eval(model, tokenizer, problems, k=k)
    base_metrics = summarize(base_records, k=k)
    base_out = Path(OUT_DIR) / f"baseline_results{tag}.json"
    base_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"metrics": base_metrics, "records": base_records},
              open(base_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[阶段1] 基线完成 → {base_out}")
    del model, tokenizer

    # 2) 训练
    print("\n════ 阶段 2/4 · GRPO 训练 ════")
    train_main(max_steps=max_steps, lr=lr, tag=tag)

    # 3) 训练后：同一批问题再答一次
    print("\n════ 阶段 3/4 · 训练后评估 ════")
    ckpt = Path(OUT_DIR) / f"grpo_ckpt{('_'+tag) if tag else ''}"
    model, tokenizer = load_model_and_tokenizer(str(ckpt))
    post_records = run_eval(model, tokenizer, problems, k=k)
    post_metrics = summarize(post_records, k=k)
    post_out = Path(OUT_DIR) / f"post_train_results{tag}.json"
    json.dump({"metrics": post_metrics, "records": post_records},
              open(post_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[阶段3] 训练后完成 → {post_out}")
    del model, tokenizer

    # 4) 对比报告
    print("\n════ 阶段 4/4 · 生成对比报告 ════")
    write_report(
        {"metrics": base_metrics, "records": base_records,
         "metric_by_level": metric_by_level(base_records)},
        {"metrics": post_metrics, "records": post_records,
         "metric_by_level": metric_by_level(post_records)},
        report_path=Path(OUT_DIR) / f"compare_report{tag}.md",
    )
    print("\n[全部完成] 已保存：基线/训练后结果 json + compare_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--n_prompts", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--pipeline", action="store_true",
                        help="全流程：训练前评估→训练→训练后评估→对比报告")
    parser.add_argument("--n_eval", type=int, default=50, help="评估问题数")
    parser.add_argument("--eval", type=str, default="",
                        help="仅评估指定 checkpoint（写结果 json）")
    args = parser.parse_args()

    if args.pipeline:
        pipeline(args.max_steps, n_eval=args.n_eval, lr=args.lr, tag=args.tag)
    elif args.eval:
        evaluate(args.eval, out=Path(OUT_DIR) / "single_eval.json", n=args.n_eval)
    else:
        train_main(args.max_steps, args.n_prompts, args.lr, args.tag)