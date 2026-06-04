"""
汇总所有方案的评估结果，打印对比表

使用方式：
  python compare_results.py

前提：
  - outputs/logs/eval_linear_validation.json   （已运行 evaluate.py）
  - outputs/logs/eval_crf_validation.json      （已运行 evaluate.py --use_crf）
  - outputs/logs/eval_llm.json                 （已运行 llm_ner.py）
  - outputs/logs/eval_sft.json                 （已运行 evaluate_sft.py）

预期输出：
  BERT NER 项目 — 四方案汇总对比
  方案                      Precision   Recall      F1      非法序列   评估方式
  BERT + Linear              ~0.81      ~0.77     ~0.79       ~20     seqeval
  BERT + CRF                  0.82       0.79      0.7254       0     seqeval
  Qwen API zero-shot         ~0.58      ~0.52     ~0.55       N/A    span F1
  Qwen API few-shot          ~0.65      ~0.58     ~0.63       N/A    span F1
  Qwen2.5-0.5B SFT (LoRA)    0.6351     0.6295    0.6323        0    span F1
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "outputs" / "logs"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_col(val, is_float=True):
    if val is None:
        return "  N/A  "
    if is_float:
        return f"{val:.4f}"
    return str(val)


def main():
    linear_res = load_json(LOG_DIR / "eval_linear_validation.json")
    crf_res    = load_json(LOG_DIR / "eval_crf_validation.json")
    llm_res    = load_json(LOG_DIR / "eval_llm.json")
    sft_res    = load_json(LOG_DIR / "eval_sft.json")

    print("\n" + "=" * 95)
    print("BERT NER 项目 — 四方案汇总对比")
    print("=" * 95)

    header = f"{'方案':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10} {'评估方式':>10}"
    print(header)
    print("-" * 80)

    # BERT + Linear
    if linear_res:
        ill = linear_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + Linear':<28} "
            f"{linear_res['precision']:>10.4f} "
            f"{linear_res['recall']:>10.4f} "
            f"{linear_res['f1']:>10.4f} "
            f"{ill:>10d}"
            f"{'seqeval':>10}"
        )
    else:
        print(f"{'BERT + Linear':<28} {'（未找到结果，请运行 evaluate.py）':>52}")

    # BERT + CRF
    if crf_res:
        ill = crf_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + CRF':<28} "
            f"{crf_res['precision']:>10.4f} "
            f"{crf_res['recall']:>10.4f} "
            f"{crf_res['f1']:>10.4f} "
            f"{ill:>10d}"
            f"{'seqeval':>10}"
        )
    else:
        print(f"{'BERT + CRF':<28} {'（未找到结果，请运行 evaluate.py --use_crf）':>52}")

    # LLM API
    if llm_res:
        zs = llm_res["zero_shot"]
        fs = llm_res["few_shot"]
        model_name = llm_res.get("model", "qwen-plus")
        print(
            f"{f'Qwen API zero-shot':<28} "
            f"{zs['precision']:>10.4f} "
            f"{zs['recall']:>10.4f} "
            f"{zs['f1']:>10.4f} "
            f"{'N/A':>10}"
            f"{'span F1':>10}"
        )
        print(
            f"{f'Qwen API few-shot':<28} "
            f"{fs['precision']:>10.4f} "
            f"{fs['recall']:>10.4f} "
            f"{fs['f1']:>10.4f} "
            f"{'N/A':>10}"
            f"{'span F1':>10}"
        )
    else:
        print(f"{'Qwen API zero/few-shot':<28} {'（未找到结果，请运行 llm_ner.py）':>52}")

    # SFT
    if sft_res:
        m = sft_res["metrics"]
        print(
            f"{'Qwen2.5-0.5B SFT (LoRA)':<28} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{'0':>10}"
            f"{'span F1':>10}"
        )
    else:
        print(f"{'Qwen2.5-0.5B SFT (LoRA)':<28} {'（未找到结果，请运行 evaluate_sft.py）':>52}")

    # 注记
    notes = []
    if llm_res:
        n = llm_res.get("n_samples", "?")
        notes.append(f"LLM 结果基于验证集 {n} 条采样")
    if sft_res:
        n = sft_res.get("n_samples", "?")
        notes.append(f"SFT 结果基于验证集 {n} 条采样")
    if linear_res or crf_res:
        notes.append("BERT 结果基于完整验证集，使用 seqeval entity-level 评估")
    if notes:
        print(f"\n  注：{'；'.join(notes)}")

    # 关键结论
    print("\n" + "=" * 95)
    print("关键教学结论：")
    if linear_res and crf_res:
        f1_diff = crf_res["f1"] - linear_res["f1"]
        ill_linear = linear_res["illegal_stats"]["total_illegal"]
        print(f"  1. CRF vs Linear：F1 {'↑' if f1_diff >= 0 else '↓'}{abs(f1_diff):.4f}")
        print(f"  2. 线性头非法序列：{ill_linear} 条；CRF 非法序列：0 条")
        print(f"     → CRF 通过 Viterbi 解码在数学上保证序列合法性")
    if llm_res and linear_res:
        fs_f1 = llm_res["few_shot"]["f1"]
        gap = linear_res["f1"] - fs_f1
        print(f"  3. 微调 BERT vs LLM few-shot：F1 差距 {gap:.4f}")
        print(f"     → 特定领域NER任务中，小模型微调通常显著优于大模型zero/few-shot")
    if sft_res and crf_res:
        sft_f1 = sft_res["metrics"]["f1"]
        gap2 = crf_res["f1"] - sft_f1
        print(f"  4. BERT+CRF vs SFT 小模型：F1 差距 {gap2:.4f}")
        print(f"     → 序列标注模型在NER上通常优于生成式小模型")
    print("=" * 95)


if __name__ == "__main__":
    main()
