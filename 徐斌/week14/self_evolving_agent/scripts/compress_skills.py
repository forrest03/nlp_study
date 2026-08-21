#!/usr/bin/env python3
"""压缩当前 Skills 体积后做一次全量评测（无 Nudge）。"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from agent import TopicMatchAgent  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from llm_config import current_model_info, get_chat_client  # noqa: E402
from skill_manager import SkillManager  # noqa: E402

SKILLS_DIR = ROOT / "skills"
VERSIONS_DIR = ROOT / "outputs" / "skill_versions"
BACKUP_DIR = ROOT / "outputs" / "skills_before_compress"
EVAL_SET = ROOT / "data" / "eval_set.json"
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"
EVOLUTION_LOG = ROOT / "outputs" / "evolution_log.json"
COMPRESS_CMP = ROOT / "outputs" / "compress_comparison.json"

COMPRESS_PROMPT = """你是 Skill 文档压缩助手。请把下面的 SKILL.md 改写成更省 token 的版本。

硬性要求：
1. 保留 YAML frontmatter（name / description / version），version 数字 +1。
2. 「输出名称（必须一字不差）」中的知识点名称必须与原文完全一致，一字不差。
3. 保留全部关键触发特征（可更短表述，不可删关键条件）。
4. 保留全部反混淆 / 反例规则（可更短，不可删关键边界）。
5. 删除冗长示例、重复说明、HTML 注释、无关段落。
6. 正文控制在约 40 行以内，尽量短；多用短 bullet。
7. 只输出完整 SKILL.md（含 frontmatter），不要解释。

当前 Skill 名：{name}

原文：
```
{content}
```
"""


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:markdown|md|yaml)?\s*\n([\s\S]*?)\n```\s*$", text)
    if m:
        return m.group(1).strip()
    return text


def compress_one(client, model: str, name: str, content: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": COMPRESS_PROMPT.format(name=name, content=content),
            }
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    out = _strip_fence(resp.choices[0].message.content or "")
    if "---" not in out or "输出名称" not in out:
        raise ValueError(f"压缩结果格式异常: {name}")
    return out


def backup_skills() -> None:
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(SKILLS_DIR, BACKUP_DIR)
    print(f"✓ 已备份当前 Skills → {BACKUP_DIR.relative_to(ROOT)}")


def run_full_eval(agent: TopicMatchAgent, evaluator: Evaluator, sm: SkillManager) -> dict:
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    total, correct = 0, 0
    by_category: dict = {}
    answers: dict = {}
    usage_sum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for qid in sorted(evaluator.questions.keys()):
        q = evaluator.questions[qid]
        answer = agent.answer(q["question"])
        ok, reason = evaluator.evaluate_answer(answer, qid)
        total += 1
        cat = q["category"]
        by_category.setdefault(cat, {"total": 0, "correct": 0})
        by_category[cat]["total"] += 1
        if ok:
            correct += 1
            by_category[cat]["correct"] += 1
        answers[str(qid)] = {
            "answer": answer,
            "correct": ok,
            "fail_reason": reason if not ok else "",
            "usage": agent.last_usage,
        }
        for k in usage_sum:
            usage_sum[k] += agent.last_usage.get(k, 0)
        mark = "✓" if ok else "✗"
        print(f"  {mark} Q{qid} → {answer[:40]}")

    for cat in by_category.values():
        cat["accuracy"] = round(cat["correct"] / cat["total"], 3)

    result = {
        "run_id": "after_compress",
        "label": "压缩后全量评测（无 Nudge）",
        "timestamp": datetime.now().isoformat(),
        "skill_versions_active": sm.get_active_versions(),
        "skill_token_stats": sm.estimate_tokens(),
        "usage_sum": usage_sum,
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 3),
        },
        "by_category": by_category,
        "answers": answers,
    }
    out = EVAL_RUNS_DIR / "after_compress.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 已写入 {out.relative_to(ROOT)}")
    return result


def update_logs(before_tokens: int, after_tokens: int, before_acc: float, after_acc: float) -> None:
    cmp = {
        "before_compress_accuracy": before_acc,
        "after_compress_accuracy": after_acc,
        "before_compress_skill_tokens": before_tokens,
        "after_compress_skill_tokens": after_tokens,
        "accuracy_delta": round(after_acc - before_acc, 3),
        "skill_tokens_delta": after_tokens - before_tokens,
        "timestamp": datetime.now().isoformat(),
    }
    COMPRESS_CMP.write_text(json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 已写入 {COMPRESS_CMP.relative_to(ROOT)}")

    if EVOLUTION_LOG.exists():
        log = json.loads(EVOLUTION_LOG.read_text(encoding="utf-8"))
        comparison = log.setdefault("comparison", {})
        comparison["compress_before_accuracy"] = before_acc
        comparison["compress_after_accuracy"] = after_acc
        comparison["compress_before_skill_tokens"] = before_tokens
        comparison["compress_after_skill_tokens"] = after_tokens
        comparison["compress_accuracy_delta"] = cmp["accuracy_delta"]
        comparison["compress_skill_tokens_delta"] = cmp["skill_tokens_delta"]
        EVOLUTION_LOG.write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✓ 已更新 {EVOLUTION_LOG.relative_to(ROOT)} comparison（保留历史）")


def main() -> None:
    info = current_model_info()
    print("=" * 60)
    print("Skill 压缩 + 全量评测")
    print(f"模型: {info['display']} ({info['provider']}/{info['model']})")
    print("=" * 60)

    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))
    skills = sm.load_all()
    if not skills:
        raise SystemExit("skills/ 为空，无法压缩")

    before_stats = sm.estimate_tokens()
    before_tokens = before_stats["total_est_tokens"]
    # 进化后准确率作为压缩前基准（来自 evolution_log / final eval）
    before_acc = 0.781
    if EVOLUTION_LOG.exists():
        log = json.loads(EVOLUTION_LOG.read_text(encoding="utf-8"))
        before_acc = float(log.get("comparison", {}).get("final_accuracy", before_acc))
    final_eval = ROOT / "outputs" / "eval_runs" / "final.json"
    if final_eval.exists():
        before_acc = float(
            json.loads(final_eval.read_text(encoding="utf-8"))["summary"]["accuracy"]
        )

    print(f"压缩前: {before_stats['count']} skills, ≈{before_tokens} est_tokens, acc={before_acc:.1%}")

    backup_skills()

    client, model = get_chat_client()
    for name, content in sorted(skills.items()):
        print(f"\n── 压缩 {name}（{len(content)} chars）──")
        try:
            compressed = compress_one(client, model, name, content)
        except Exception as e:
            print(f"  ✗ LLM 压缩失败，保留原文: {e}")
            continue
        lines = compressed.count("\n") + 1
        path = SKILLS_DIR / name / "SKILL.md"
        path.write_text(compressed + ("\n" if not compressed.endswith("\n") else ""), encoding="utf-8")
        sm._save_version(name, compressed, action="compress", reason="token-efficient rewrite")
        print(f"  ✓ {len(content)} → {len(compressed)} chars, {lines} lines")

    after_stats = sm.estimate_tokens()
    after_tokens = after_stats["total_est_tokens"]
    print(f"\n压缩后 Skill 体积: ≈{before_tokens} → ≈{after_tokens} est_tokens")

    print("\n" + "─" * 60)
    print("全量评测（压缩后，无 Nudge）")
    print("─" * 60)
    evaluator = Evaluator(str(EVAL_SET))
    agent = TopicMatchAgent(sm, nudge_interval=0)
    result = run_full_eval(agent, evaluator, sm)
    evaluator.print_report(
        {**result["summary"], "by_category": result["by_category"], "errors": []},
        "压缩后",
    )

    after_acc = result["summary"]["accuracy"]
    print("\n" + "=" * 60)
    print("压缩前后对比")
    print("=" * 60)
    print(f"  准确率:     {before_acc:.1%} → {after_acc:.1%}  (Δ {after_acc - before_acc:+.1%})")
    print(f"  Skill体积:  {before_tokens} → {after_tokens} est_tokens")

    update_logs(before_tokens, after_tokens, before_acc, after_acc)


if __name__ == "__main__":
    main()
