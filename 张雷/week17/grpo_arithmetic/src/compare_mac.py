"""
Mac 训练前后对比：读取 Mac 实测的基线/训练后 probe + 训练日志，
复用 compare_results.py 的表格与绘图函数，生成对比表与训练曲线。

与 compare_results.py 的区别：指向 Mac 的独立文件（--tag mac_full 系列），
不覆盖原 CUDA 实验数据（baseline_probe.json / post_train_probe.json 等）。

使用方式：
  python src/compare_mac.py

输入（均在 outputs/ 下）：
  baseline_probe_mac_full.json      # Mac 基线（n=50, seed=42）
  post_train_probe_mac_full.json    # Mac 全量训练后（同 seed=42）
  train_log_mac_full.json           # Mac 全量训练日志
  train_log_lora_mac_test.json      # （可选）Mac LoRA 短训日志，叠加曲线

输出：
  outputs/figures/train_curves_mac.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from compare_results import OUT, fmt_examples, fmt_table, plot_curves  # noqa: E402


def load(name: str):
    p = OUT / name
    if not p.exists():
        raise FileNotFoundError(f"缺少文件：{p}，请先运行对应的 probe/训练脚本")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    base = load("baseline_probe_mac_full.json")
    post = load("post_train_probe_mac_full.json")
    log_full = load("train_log_mac_full.json")

    reports = [("Mac基线", base), ("Mac全量", post)]
    log_entries = [("full(mac)", log_full)]

    # 可选：叠加今天早些时候的 Mac LoRA 短训曲线
    lora_log_path = OUT / "train_log_lora_mac_test.json"
    if lora_log_path.exists():
        with open(lora_log_path, encoding="utf-8") as f:
            log_entries.append(("lora(mac,50step)", json.load(f)))

    print("=" * 96)
    print("Mac 训练前后对比（同一评估集，seed=42，50 题/难度；格式率 / greedy正确率 / pass@8）")
    print("=" * 96)
    print(fmt_table(reports))
    print("\n" + "=" * 96)
    print("样例对照（greedy 解码，Mac基线 vs Mac训练后）")
    print(fmt_examples(base, post))

    fig_path = OUT / "figures" / "train_curves_mac.png"
    plot_curves(log_entries, fig_path)
    print(f"\n对比完成。曲线图：{fig_path}")


if __name__ == "__main__":
    main()
