"""
演示主程序：习题知识点匹配 Skill 自进化实验。

用法：
  cd self_evolving_agent
  export LLM_PROVIDER=qwen
  export DASHSCOPE_API_KEY=sk-xxx
  python scripts/build_eval_from_dataset.py
  python src/demo_runner.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import TopicMatchAgent  # noqa: E402
from background_reviewer import BackgroundReviewer  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from llm_config import current_model_info  # noqa: E402
from skill_manager import SkillManager  # noqa: E402

SKILLS_DIR = ROOT / "skills"
SKILLS_ORIG = ROOT / "outputs" / "skills_original"
EVAL_SET = ROOT / "data" / "eval_set.json"
DEMO_SCRIPT = ROOT / "data" / "demo_script.json"
POLICIES = ROOT / "data" / "policies.md"
VERSIONS_DIR = ROOT / "outputs" / "skill_versions"
EVAL_RUNS_DIR = ROOT / "outputs" / "eval_runs"
EVOL_LOG = ROOT / "outputs" / "evolution_log.json"


def ensure_original(sm: SkillManager) -> None:
    if not SKILLS_ORIG.exists():
        shutil.copytree(SKILLS_DIR, SKILLS_ORIG)
        print(f"✓ 首次运行：原始 Skills 备份至 {SKILLS_ORIG.name}/")
        for skill_name, content in sm.load_all().items():
            sm._save_version(skill_name, content, action="initial", reason="初始版本")
    else:
        print("✓ 检测到原始备份，已跳过覆盖")


def restore_from_original() -> None:
    if not SKILLS_ORIG.exists():
        raise RuntimeError("原始备份不存在，请删除 outputs/skills_original 外的产物后重跑")
    if SKILLS_DIR.exists():
        shutil.rmtree(SKILLS_DIR)
    shutil.copytree(SKILLS_ORIG, SKILLS_DIR)
    for path in [
        VERSIONS_DIR,
        ROOT / "outputs" / "skill_snapshots",
        EVAL_RUNS_DIR,
    ]:
        if path.exists():
            shutil.rmtree(path)
    print("✓ 已还原初始 Skills，清空上次版本历史")


def run_probe_eval(
    agent: TopicMatchAgent,
    evaluator: Evaluator,
    probe_ids: list[int],
    run_id: str,
    label: str,
    sm: SkillManager,
) -> dict:
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    total, correct = 0, 0
    by_category: dict = {}
    answers: dict = {}
    usage_sum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for qid in probe_ids:
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

    for cat in by_category.values():
        cat["accuracy"] = round(cat["correct"] / cat["total"], 3)

    result = {
        "run_id": run_id,
        "label": label,
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
    (EVAL_RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_full_eval(
    agent: TopicMatchAgent,
    evaluator: Evaluator,
    run_id: str,
    label: str,
    sm: SkillManager,
) -> dict:
    return run_probe_eval(
        agent, evaluator, list(evaluator.questions.keys()), run_id, label, sm
    )


class EvolutionLog:
    def __init__(self) -> None:
        self.eval_runs: list[dict] = []
        self.nudge_events: list[dict] = []
        self.question_history: dict[str, list] = {}

    def add_eval_run(self, result: dict) -> None:
        self.eval_runs.append(
            {
                "run_id": result["run_id"],
                "label": result["label"],
                "timestamp": result["timestamp"],
                "skill_versions_active": result["skill_versions_active"],
                "skill_token_stats": result.get("skill_token_stats", {}),
                "usage_sum": result.get("usage_sum", {}),
                "accuracy": result["summary"]["accuracy"],
                "correct": result["summary"]["correct"],
                "total": result["summary"]["total"],
                "by_category": {
                    k: v["accuracy"] for k, v in result["by_category"].items()
                },
            }
        )
        for qid_str, ans_data in result["answers"].items():
            self.question_history.setdefault(qid_str, []).append(
                {
                    "run_id": result["run_id"],
                    "label": result["label"],
                    "skill_versions": result["skill_versions_active"],
                    "answer": ans_data["answer"],
                    "correct": ans_data["correct"],
                    "fail_reason": ans_data.get("fail_reason", ""),
                }
            )

    def add_nudge_event(
        self,
        seq: int,
        block: str,
        actions_taken: list[dict],
        accuracy_before: float,
        skill_versions_after: dict,
        skill_tokens_after: dict,
        analysis: str,
    ) -> None:
        self.nudge_events.append(
            {
                "after_seq": seq,
                "block": block,
                "timestamp": datetime.now().isoformat(),
                "accuracy_before_this_block": round(accuracy_before, 3),
                "analysis": analysis,
                "actions_taken": actions_taken,
                "skill_versions_after": skill_versions_after,
                "skill_tokens_after": skill_tokens_after,
            }
        )

    def save(self, sm: SkillManager, evaluator: Evaluator) -> None:
        skill_snapshots = {}
        if SKILLS_DIR.exists():
            for skill_dir in SKILLS_DIR.iterdir():
                if skill_dir.is_dir():
                    name = skill_dir.name
                    history = sm.get_version_history(name)
                    skill_snapshots[name] = [
                        {
                            "version": h["version"],
                            "time": h["time"],
                            "action": h["action"],
                            "reason": h["reason"][:120],
                            "est_tokens": h.get("est_tokens"),
                            "snapshot_file": h.get("snapshot_file", ""),
                        }
                        for h in history
                    ]

        question_comparison = {}
        for qid_str, history in self.question_history.items():
            qid = int(qid_str)
            q = evaluator.questions.get(qid)
            if q:
                question_comparison[qid_str] = {
                    "question": q["question"],
                    "category": q["category"],
                    "topic_name": q.get("topic_name"),
                    "ground_truth": q["ground_truth"],
                    "history": history,
                }

        baseline = next((r for r in self.eval_runs if r["run_id"] == "baseline"), None)
        final = next((r for r in self.eval_runs if r["run_id"] == "final"), None)

        log = {
            "generated_at": datetime.now().isoformat(),
            "model": current_model_info(),
            "comparison": {
                "baseline_accuracy": baseline["accuracy"] if baseline else None,
                "final_accuracy": final["accuracy"] if final else None,
                "baseline_skill_tokens": (
                    baseline.get("skill_token_stats", {}).get("total_est_tokens")
                    if baseline
                    else None
                ),
                "final_skill_tokens": (
                    final.get("skill_token_stats", {}).get("total_est_tokens")
                    if final
                    else None
                ),
                "baseline_prompt_tokens": (
                    baseline.get("usage_sum", {}).get("prompt_tokens") if baseline else None
                ),
                "final_prompt_tokens": (
                    final.get("usage_sum", {}).get("prompt_tokens") if final else None
                ),
            },
            "eval_runs": self.eval_runs,
            "nudge_events": self.nudge_events,
            "skill_snapshots": skill_snapshots,
            "question_comparison": question_comparison,
        }
        EVOL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✓ 已写入 {EVOL_LOG}")


def apply_actions(sm: SkillManager, actions: list[dict]) -> list[dict]:
    taken = []
    for act in actions:
        action = act.get("action")
        name = act.get("skill_name", "")
        reason = act.get("reason", "")
        ok = False
        if action == "create":
            ok = sm.create(name, act.get("content", ""), reason=reason)
        elif action == "patch":
            ok = sm.patch(
                name,
                act.get("old_text", ""),
                act.get("new_text", ""),
                reason=reason,
            )
        taken.append(
            {
                "action": action,
                "skill_name": name,
                "reason": reason,
                "ok": ok,
            }
        )
    return taken


def main() -> None:
    if not EVAL_SET.exists() or not DEMO_SCRIPT.exists():
        raise SystemExit(
            "缺少 data/eval_set.json 或 demo_script.json，"
            "请先运行: python scripts/build_eval_from_dataset.py"
        )

    info = current_model_info()
    print("=" * 60)
    print("习题知识点匹配 — Skill 自进化实验")
    print(f"模型: {info['display']} ({info['provider']}/{info['model']})")
    print("=" * 60)

    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))
    ensure_original(sm)
    restore_from_original()
    sm = SkillManager(str(SKILLS_DIR), str(VERSIONS_DIR))
    for name, content in sm.load_all().items():
        sm._save_version(name, content, action="initial", reason="实验起始快照")

    evaluator = Evaluator(str(EVAL_SET))
    agent = TopicMatchAgent(sm, nudge_interval=0)
    reviewer = BackgroundReviewer(str(POLICIES), sm)
    script = json.loads(DEMO_SCRIPT.read_text(encoding="utf-8"))
    questions = script["questions"]
    probe_ids = script.get("probe_question_ids", [])
    block_size = script.get("nudge_interval", 8)
    blocks = [questions[i : i + block_size] for i in range(0, len(questions), block_size)]

    evo = EvolutionLog()

    print("\n" + "─" * 60)
    print("基线评估（初始 Skills）")
    print("─" * 60)
    token0 = sm.estimate_tokens()
    print(
        f"初始 Skills: {token0['count']} 个, 约 {token0['total_est_tokens']} tokens"
    )
    baseline = run_full_eval(agent, evaluator, "baseline", "基线（初始 Skills）", sm)
    evo.add_eval_run(baseline)
    evaluator.print_report(
        {
            **baseline["summary"],
            "by_category": baseline["by_category"],
            "errors": [],
        },
        "基线",
    )

    for bi, block in enumerate(blocks, 1):
        cat = block[0]["category"] if block else "?"
        print("\n" + "━" * 60)
        print(f"第 {bi}/{len(blocks)} 块 [{cat}] 共 {len(block)} 题")
        print("━" * 60)

        failed: list[dict] = []
        correct_n = 0
        for item in block:
            qid = item["id"]
            answer = agent.answer(item["question"])
            ok, reason = evaluator.evaluate_answer(answer, qid)
            mark = "✓" if ok else "✗"
            print(f"  {mark} Q{qid} → {answer[:40]}")
            if ok:
                correct_n += 1
            else:
                failed.append(
                    {
                        "question": item["question"],
                        "answer": answer,
                        "fail_reason": reason,
                        "topic_name": item.get("topic_name")
                        or evaluator.questions[qid].get("topic_name"),
                    }
                )

        acc = correct_n / len(block) if block else 0
        print(f"  本块完成: {correct_n}/{len(block)} = {acc:.1%}")

        if not failed:
            print("  ✓ 本块全对，跳过 Nudge 和 Probe")
            continue

        print(f"  🔔 Nudge 触发（{len(failed)} 条失败样本）")
        actions = reviewer.review(failed)
        taken = apply_actions(sm, actions)
        print(f"  ✓ 执行了 {sum(1 for t in taken if t['ok'])} 个 Skill 操作")
        token_after = sm.estimate_tokens()
        evo.add_nudge_event(
            seq=bi,
            block=cat,
            actions_taken=taken,
            accuracy_before=acc,
            skill_versions_after=sm.get_active_versions(),
            skill_tokens_after=token_after,
            analysis=reviewer.last_analysis,
        )

        probe = run_probe_eval(
            agent,
            evaluator,
            probe_ids,
            f"probe_after_block_{bi}",
            f"Probe after block {bi} [{cat}]",
            sm,
        )
        evo.add_eval_run(probe)
        print(
            f"  Probe: {probe['summary']['correct']}/{probe['summary']['total']} "
            f"= {probe['summary']['accuracy']:.1%} | "
            f"Skills≈{token_after['total_est_tokens']} tokens"
        )

    print("\n" + "─" * 60)
    print("最终评估")
    print("─" * 60)
    final = run_full_eval(agent, evaluator, "final", "最终（进化后 Skills）", sm)
    evo.add_eval_run(final)
    evaluator.print_report(
        {
            **final["summary"],
            "by_category": final["by_category"],
            "errors": [],
        },
        "最终",
    )

    print("\n" + "=" * 60)
    print("优化前后对比")
    print("=" * 60)
    b_acc = baseline["summary"]["accuracy"]
    f_acc = final["summary"]["accuracy"]
    b_tok = baseline["skill_token_stats"]["total_est_tokens"]
    f_tok = final["skill_token_stats"]["total_est_tokens"]
    b_pt = baseline["usage_sum"]["prompt_tokens"]
    f_pt = final["usage_sum"]["prompt_tokens"]
    print(f"  准确率:     {b_acc:.1%} → {f_acc:.1%}  (Δ {(f_acc - b_acc):+.1%})")
    print(f"  Skill体积:  {b_tok} → {f_tok} est_tokens")
    print(f"  评测Prompt: {b_pt} → {f_pt} tokens")
    print(f"  最终 Skills: {list(sm.load_all().keys())}")

    evo.save(sm, evaluator)


if __name__ == "__main__":
    main()
