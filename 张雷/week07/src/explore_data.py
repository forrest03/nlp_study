"""
人民日报 NER 数据集探索与可视化（BIO 标注格式）

教学重点：
  1. BIO 标注格式的解析方法（B-实体名 / I-实体名 / O）
  2. 各实体类型的分布差异（为什么类别不均衡是NER的难点）
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布（短实体 vs 长实体的识别难度差异）

使用方式：
  python explore_data.py

输出：
  outputs/figures/entity_distribution.png      各类实体频次直方图
  outputs/figures/text_length_distribution.png  文本长度分布（含P95线）
  outputs/figures/entity_length_distribution.png 实体字符数分布
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs" / "figures"

# BIO 标签 → 实体类型映射
BIO_TO_ENTITY = {
    "B-PER": "PER", "I-PER": "PER",
    "B-ORG": "ORG", "I-ORG": "ORG",
    "B-LOC": "LOC", "I-LOC": "LOC",
}


def load_split(split: str) -> list:
    path = DATA_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_bio_entities(tokens: list, ner_tags: list) -> list:
    """从 BIO 标注中解析实体列表，返回 [(实体文本, 实体类型, 长度), ...]"""
    entities = []
    current_tokens = []
    current_type = None

    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            # 保存上一个实体
            if current_tokens and current_type:
                entities.append(("".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = tag[2:]  # 去掉 "B-" 前缀
        elif tag.startswith("I-"):
            if current_tokens and tag[2:] == current_type:
                current_tokens.append(token)
            else:
                # 孤立的 I- 标签，当作新实体开始
                if current_tokens and current_type:
                    entities.append(("".join(current_tokens), current_type))
                current_tokens = [token]
                current_type = tag[2:] if len(tag) > 2 else None
        else:  # O 标签
            if current_tokens and current_type:
                entities.append(("".join(current_tokens), current_type))
            current_tokens = []
            current_type = None

    # 处理最后一个实体
    if current_tokens and current_type:
        entities.append(("".join(current_tokens), current_type))

    return entities


def collect_stats(records: list) -> dict:
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}

    for row in records:
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        text_lengths.append(len(tokens))

        entities = parse_bio_entities(tokens, ner_tags)
        entity_per_sentence.append(len(entities))

        for surface, etype in entities:
            entity_type_counts[etype] += 1
            entity_lengths.append(len(surface))
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(surface)

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def print_summary(stats_train: dict, stats_val: dict):
    print("=" * 70)
    print("人民日报 NER 数据集统计摘要（BIO 标注格式）")
    print("=" * 70)

    n_train = len(stats_train['text_lengths'])
    n_val = len(stats_val['text_lengths'])

    print(f"\n【数据规模】")
    print(f"  训练集：{n_train} 条")
    print(f"  验证集：{n_val} 条")
    print(f"  标签类别：PER(人名) / ORG(组织机构) / LOC(地名)")

    print("\n【训练集】")
    print(f"  样本数：{n_train} 条")
    print(f"  文本平均长度：{sum(stats_train['text_lengths']) / n_train:.1f} 字")
    print(f"  文本最大长度：{max(stats_train['text_lengths'])} 字")
    print(f"  文本长度中位数：{sorted(stats_train['text_lengths'])[n_train // 2]} 字")
    print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / n_train:.2f}")
    print(f"  实体总数：{sum(stats_train['entity_type_counts'].values())}")
    print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} 字" if stats_train['entity_lengths'] else "  平均实体长度：N/A")

    print("\n【验证集】")
    print(f"  样本数：{n_val} 条")
    print(f"  文本平均长度：{sum(stats_val['text_lengths']) / n_val:.1f} 字")
    print(f"  平均实体数/句：{sum(stats_val['entity_per_sentence']) / n_val:.2f}")
    print(f"  实体总数：{sum(stats_val['entity_type_counts'].values())}")

    print("\n【各类实体频次（训练集）】")
    et_label = {"PER": "人名", "ORG": "组织机构", "LOC": "地名"}
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = et_label.get(etype, etype)
        print(f"  {etype:6s} ({cn:8s}) : {cnt:5d} 条")

    print("\n【各类实体示例（训练集，取前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = et_label.get(etype, etype)
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:6s} ({cn}) : {' | '.join(examples)}")

    print()


def plot_entity_distribution(stats_train: dict):
    """各类实体频次直方图"""
    et_label = {"PER": "人名", "ORG": "组织机构", "LOC": "地名"}
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({et_label.get(k, k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(labels, values, color=colors[:len(labels)], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("人民日报 NER 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "entity_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_distribution.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict):
    """文本长度分布（含 P95 线）"""
    lengths = stats_train["text_lengths"]
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    p99 = sorted(lengths)[int(len(lengths) * 0.99)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title("人民日报 文本长度分布（训练集）", fontsize=14)
    ax.set_xlabel("文本字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "text_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'text_length_distribution.png'}")
    plt.close()
    print(f"  P95 文本长度={p95}，P99={p99}，建议 max_length=128")


def plot_entity_length_distribution(stats_train: dict):
    """实体字符数分布"""
    lengths = Counter(stats_train["entity_lengths"])
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:25]], ys[:25], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title("人民日报 实体长度分布（训练集，前25）", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "entity_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_length_distribution.png'}")
    plt.close()

    avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
    print(f"  实体平均长度={avg_len:.1f}字，CRF 对短实体边界识别优势更明显")


def main():
    parse_args()

    train_records = load_split("train")
    val_records = load_split("validation")

    stats_train = collect_stats(train_records)
    stats_val = collect_stats(val_records)

    print_summary(stats_train, stats_val)

    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)

    print("\n探索完成！图表已保存到 outputs/figures/")
    print("下一步：python train.py               # 训练 BERT+Linear")
    print("         python train.py --use_crf    # 训练 BERT+CRF")


def parse_args():
    parser = argparse.ArgumentParser(description="探索人民日报 NER 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()
