"""
训练前后对比（扩展版）：基线 vs 全量 vs LoRA（含 L1~L10 全部 10 个难度）

教学重点：
  1. 新旧难度对比：训练集内难度（L3/L5/L8/L9/L10）vs 未训练难度（L1/L2/L4/L6/L7）
  2. 灾难性遗忘检测：训练集外的老难度是否严重退化
  3. 泛化能力证据：未训练的新难度（L7 小数）是否也被 RL 间接学会
  4. 训练曲线解读：扩展难度下奖励曲线是否仍呈"格式先收敛、正确率后爬坡"

使用方式：
  python src/compare_results.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

# 全部 10 个难度（扩展版）
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

# 训练集内难度（本作业的默认配置 + --include_l7 时的扩展）
DEFAULT_TRAINED = {
    "L3_addsub_3digit",
    "L5_mul_2x1digit",
    "L8_word_problem",
    "L9_paren_mixed",
    "L10_chain_3num",
}
EXTENDED_TRAINED = DEFAULT_TRAINED | {"L7_decimal_addsub"}


def fmt_table(reports):
    """reports: [(标签, probe_report_dict), ...]，第一个是基线。"""
    base = reports[0][1]
    header = f"{'难度':<22}{'训练集':^6}"
    for name, _ in reports:
        header += f"{name + ' 格式/正确/pass@8':^30}"
    rows = []
    for lv in LEVELS:
        trained = "√" if lv in DEFAULT_TRAINED else "—"
        if lv in EXTENDED_TRAINED and lv not in DEFAULT_TRAINED:
            trained = "+"   # --include_l7 时才训
        row = f"{lv:<22}{trained:^6}"
        for name, rep in reports:
            r = rep[lv]
            row += (
                f"{r['greedy_format_rate']:.2f} / {r['greedy_loose_acc']:.2f} / {r['loose_pass@8']:.2f}"
                .center(30)
            )
        rows.append(row)
    return header + "\n" + "\n".join(rows)


def fmt_examples(base, post, n=2):
    """训练集内每个新难度各取 n 条 greedy 输出对照（旧项目只对比 L2/L3/L5）。"""
    lines = []
    trained_levels = ["L3_addsub_3digit", "L5_mul_2x1digit",
                       "L8_word_problem", "L9_paren_mixed", "L10_chain_3num"]
    for lv in trained_levels:
        if lv not in base or lv not in post:
            continue
        lines.append(f"\n--- {lv} ---")
        # 兼容老 probe 数据（没有该难度时跳过）
        examples_b = base[lv].get("examples", [])
        examples_p = post[lv].get("examples", [])
        for i in range(min(n, len(examples_b), len(examples_p))):
            eb, ep = examples_b[i], examples_p[i]
            # 应用题 prompt 较长，截断显示
            expr_disp = eb["expr"][:60] + "..." if len(eb["expr"]) > 60 else eb["expr"]
            lines.append(f"  {expr_disp} = {eb['answer']}")
            lines.append(f"    前: {eb['greedy_output'][:80]!r}")
            lines.append(f"    后: {ep['greedy_output'][:80]!r}")
    return "\n".join(lines)


def plot_curves(log_entries, fig_path):
    """log_entries: [(标签, log_history), ...]，多条曲线叠加对比。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, log_history in log_entries:
        logs = [e for e in log_history if "reward" in e]
        steps = [e["step"] for e in logs]
        axes[0].plot(steps, [e["rewards/reward_correct/mean"] for e in logs],
                     label=f"{name} correct")
        axes[0].plot(steps, [e["rewards/reward_format/mean"] for e in logs],
                     linestyle="--", label=f"{name} format")
        axes[1].plot(steps, [e["frac_reward_zero_std"] for e in logs], label=name)
        axes[2].plot(steps, [e["entropy"] for e in logs], label=name)

    axes[0].set_title("Reward components (group mean)")
    axes[0].set_xlabel("step")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("frac_reward_zero_std\n(degenerate group ratio)")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].set_title("Policy entropy")
    axes[2].set_xlabel("step")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"训练曲线已保存：{fig_path}")


def main():
    base_path = OUT / "baseline_probe.json"
    post_full_path = OUT / "post_train_probe.json"
    log_full_path = OUT / "train_log.json"

    if not base_path.exists():
        print(f"[错误] 缺少基线数据 {base_path}，请先跑 probe_baseline.py")
        return
    if not post_full_path.exists():
        print(f"[错误] 缺少训练后数据 {post_full_path}，请先跑 probe_baseline.py --model")
        return

    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(post_full_path, encoding="utf-8") as f:
        post_full = json.load(f)

    reports = [("基线", base), ("全量", post_full)]
    log_entries = [("full", [])]
    if log_full_path.exists():
        with open(log_full_path, encoding="utf-8") as f:
            log_entries = [("full", json.load(f))]

    # LoRA 实验存在时纳入三方对比
    lora_probe_path = OUT / "post_train_probe_lora.json"
    lora_log_path = OUT / "train_log_lora.json"
    if lora_probe_path.exists():
        with open(lora_probe_path, encoding="utf-8") as f:
            reports.append(("LoRA", json.load(f)))
    if lora_log_path.exists():
        with open(lora_log_path, encoding="utf-8") as f:
            log_entries.append(("lora", json.load(f)))

    print("=" * 110)
    print("训练前后对比（扩展算术题，10 难度 × 50 题；格式率 / greedy正确率 / pass@8）")
    print("√ = 默认训练集； + = --include_l7 时加入； — = 留作泛化评估")
    print("=" * 110)
    print(fmt_table(reports))
    print("\n" + "=" * 110)
    print("样例对照（greedy 解码，基线 vs 全量，训练集内难度）")
    print(fmt_examples(base, post_full))

    if log_entries[0][1]:
        plot_curves(log_entries, OUT / "figures" / "train_curves.png")


if __name__ == "__main__":
    main()