"""
评估器：关键词匹配 + 「无法匹配」推脱一票否决。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

NEG_PREFIXES = ("不", "无", "非", "未", "没")
NEG_WINDOW = 4
DEFERRAL_SIGNAL = "无法匹配"


def _normalize(text: str) -> str:
    text = re.sub(r"(?<=\d)[,，](?=\d)", "", text)
    text = text.replace(" ", "").replace("\u3000", "")
    return text.lower()


def _forbidden_hits(text: str, keyword: str) -> bool:
    idx = 0
    while True:
        pos = text.find(keyword, idx)
        if pos == -1:
            return False
        prefix = text[max(0, pos - NEG_WINDOW) : pos]
        if not any(neg in prefix for neg in NEG_PREFIXES):
            return True
        idx = pos + 1


class Evaluator:
    def __init__(self, eval_set_path: str):
        data = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
        self.questions = {q["id"]: q for q in data["questions"]}

    def evaluate_answer(self, answer: str, question_id: int) -> tuple[bool, str]:
        gt = self.questions[question_id]["ground_truth"]
        ans_norm = _normalize(answer)

        if DEFERRAL_SIGNAL in ans_norm or "联系人工" in ans_norm:
            return False, f"Agent 推脱（含 '{DEFERRAL_SIGNAL}'）"

        for kw in gt.get("required", []):
            if _normalize(kw) not in ans_norm:
                return False, f"缺少关键词: '{kw}'"

        for kw in gt.get("forbidden", []):
            if _forbidden_hits(ans_norm, _normalize(kw)):
                return False, f"出现禁止词: '{kw}'"

        return True, "correct"

    def run_full_eval(self, agent_func, verbose: bool = False) -> dict:
        total = 0
        correct = 0
        by_category = defaultdict(lambda: {"total": 0, "correct": 0})
        errors = []

        for qid, q in sorted(self.questions.items()):
            answer = agent_func(q["question"])
            ok, reason = self.evaluate_answer(answer, qid)
            total += 1
            by_category[q["category"]]["total"] += 1
            if ok:
                correct += 1
                by_category[q["category"]]["correct"] += 1
            else:
                errors.append(
                    {
                        "id": qid,
                        "category": q["category"],
                        "reason": reason,
                        "question": q["question"][:40],
                    }
                )
                if verbose:
                    print(f"  ✗ Q{qid} [{q['category']}] {reason}")

        for cat in by_category.values():
            cat["accuracy"] = round(cat["correct"] / cat["total"], 3)

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 3),
            "by_category": dict(by_category),
            "errors": errors,
        }

    def print_report(self, result: dict, label: str = "") -> None:
        header = f"=== 评估报告 {label} ===" if label else "=== 评估报告 ==="
        print(f"\n{header}")
        print(f"总体准确率: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
        print("\n分类准确率:")
        for cat, stats in sorted(result["by_category"].items()):
            bar = "█" * int(stats["accuracy"] * 20)
            print(
                f"  {cat:<22} {stats['correct']:>2}/{stats['total']:>2}  "
                f"{bar} {stats['accuracy']:.0%}"
            )
        if result.get("errors"):
            print("\n错误样本（前5条）:")
            for err in result["errors"][:5]:
                print(f"  Q{err['id']} {err['question']}... → {err['reason']}")
