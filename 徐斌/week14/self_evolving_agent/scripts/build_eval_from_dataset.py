#!/usr/bin/env python3
"""从 topic_bundle.json 生成 eval_set / demo_script / policies。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def cat_of(name: str) -> str:
    mapping = [
        ("最简整数比", "ratio_simplify"),
        ("韦达", "vieta"),
        ("配方", "completing_square"),
        ("公式法", "quadratic_formula"),
        ("勾股", "pythagorean"),
        ("全等", "congruent_triangle"),
        ("分式方程", "fractional_eq"),
        ("整式代入", "polynomial_sub"),
    ]
    for key, value in mapping:
        if key in name:
            return value
    return re.sub(r"[^\w]+", "_", name)[:24]


def main() -> None:
    bundle = json.loads((DATA / "topic_bundle.json").read_text(encoding="utf-8"))

    questions: list[dict] = []
    qid = 1
    by_cat: dict[str, list[int]] = defaultdict(list)

    for topic in bundle:
        cat = cat_of(topic["topicName"])
        topic_name = topic["topicName"]
        required = [topic_name]
        forbidden = [
            other["topicName"] for other in bundle if other["topicName"] != topic_name
        ][:3]

        bodies: list[str] = []
        for skill in topic.get("skills") or []:
            for body in skill.get("representativeBodies") or []:
                body = (body or "").strip()
                if body and body not in bodies:
                    bodies.append(body)

        for i, body in enumerate(bodies[:8]):
            qtext = body if len(body) <= 400 else body[:400] + "…"
            item = {
                "id": qid,
                "category": cat,
                "topic_name": topic_name,
                "difficulty": "easy" if i < 3 else "medium",
                "question": (
                    "请判断下面这道题考查的知识点是什么？只输出知识点名称。\n\n"
                    + qtext
                ),
                "problem_body": qtext,
                "ground_truth": {
                    "required": required,
                    "forbidden": forbidden,
                    "_note": "答案须包含完整知识点名称",
                },
                "initial_skill_handles": cat in ("ratio_simplify", "vieta"),
            }
            questions.append(item)
            by_cat[cat].append(qid)
            qid += 1

    block_order = [
        "ratio_simplify",
        "vieta",
        "completing_square",
        "quadratic_formula",
        "pythagorean",
        "congruent_triangle",
        "fractional_eq",
        "polynomial_sub",
    ]

    demo_qs: list[dict] = []
    for cat in block_order:
        for qid_i in by_cat.get(cat, [])[:8]:
            q = next(x for x in questions if x["id"] == qid_i)
            demo_qs.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "topic_name": q["topic_name"],
                }
            )

    probe_ids: list[int] = []
    for cat in block_order:
        probe_ids.extend(by_cat.get(cat, [])[:3])

    eval_set = {
        "description": "初中数学习题→知识点匹配评估集",
        "questions": questions,
    }
    demo_script = {
        "nudge_interval": 8,
        "description": "按知识点类别分块演示 Skill 自进化；每块失败后触发 Reviewer",
        "probe_question_ids": probe_ids,
        "questions": demo_qs,
    }

    lines = [
        "# 知识点判定标准（仅 Reviewer 可读）\n\n",
        "本文档是习题知识点匹配的权威依据。主 Agent 不能直接访问。\n\n",
        "匹配原则：根据题干考查的核心概念/方法，输出**唯一**知识点名称"
        "（须与下列名称完全一致）。\n",
    ]
    for topic in bundle:
        cat = cat_of(topic["topicName"])
        lines.append(f"\n## {topic['topicName']}\n")
        lines.append(f"- category: `{cat}`\n")
        ctx = (topic.get("topicContext") or "").strip()
        if ctx:
            lines.append(f"- 定义与考点：{ctx}\n")
        lines.append("- 技能桶与代表题：\n")
        for skill in (topic.get("skills") or [])[:5]:
            name = skill.get("name") or ""
            lines.append(f"  - 【{name}】题量={skill.get('problemCount', 0)}\n")
            for body in (skill.get("representativeBodies") or [])[:1]:
                snippet = body.replace("\n", " ")
                if len(snippet) > 160:
                    snippet = snippet[:160] + "…"
                lines.append(f"    例：{snippet}\n")
        lines.append("- 易混淆：勿与同列表其他知识点互相替代。\n")

    (DATA / "eval_set.json").write_text(
        json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA / "demo_script.json").write_text(
        json.dumps(demo_script, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DATA / "policies.md").write_text("".join(lines), encoding="utf-8")

    print(f"questions={len(questions)} demo={len(demo_qs)} probe={len(probe_ids)}")
    print({k: len(v) for k, v in by_cat.items()})


if __name__ == "__main__":
    main()
