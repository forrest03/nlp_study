"""Compare a verbose Skill with a token-optimized Skill.

Static mode is offline and checks size plus atomic policy coverage. With --llm,
the same eight questions are sent to DeepSeek and model-reported token usage,
latency, and rule accuracy are also recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
VERSIONS = {
    "before": EXPERIMENT_DIR / "before" / "SKILL.md",
    "after": EXPERIMENT_DIR / "after" / "SKILL.md",
}
REQUIREMENTS_PATH = EXPERIMENT_DIR / "requirements.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "skill_optimization_comparison.json"

SYSTEM_TEMPLATE = """你是云购商城的智能客服助手。
只能依据给定技能回答。技能覆盖问题时直接给出结论和关键条件；确实未覆盖时仅回答“需要联系人工客服”。

## 当前技能
{skill}
"""


def estimate_tokens(text: str) -> int:
    """Stable offline estimate; real --llm runs use provider-reported usage."""
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    ascii_chunks = re.findall(r"[A-Za-z0-9_]+", text)
    ascii_tokens = sum(max(1, math.ceil(len(chunk) / 4)) for chunk in ascii_chunks)
    punctuation = len(re.findall(r"[^\w\s\u3400-\u9fff]", text))
    return cjk + ascii_tokens + math.ceil(punctuation / 2)


def coverage(skill: str, requirements: list[dict]) -> dict:
    rows = []
    for requirement in requirements:
        missing = [keyword for keyword in requirement["keywords"] if keyword not in skill]
        rows.append({
            "id": requirement["id"],
            "passed": not missing,
            "missing": missing,
        })
    passed = sum(row["passed"] for row in rows)
    return {
        "passed": passed,
        "total": len(rows),
        "rate": round(passed / len(rows), 3),
        "details": rows,
    }


def static_metrics(path: Path, requirements: list[dict]) -> dict:
    text = path.read_text(encoding="utf-8")
    tokens = estimate_tokens(text)
    covered = coverage(text, requirements)
    return {
        "path": str(path.relative_to(ROOT)),
        "characters": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "non_empty_lines": sum(bool(line.strip()) for line in text.splitlines()),
        "estimated_tokens": tokens,
        "token_method": "offline heuristic; use --llm for provider-reported usage",
        "requirement_coverage": covered,
        "covered_requirements_per_1k_tokens": round(covered["passed"] / tokens * 1000, 2),
    }


def evaluate_answer(answer: str, ground_truth: dict) -> tuple[bool, str]:
    normalized = re.sub(r"(?<=\d)[,，](?=\d)", "", answer).lower()
    if "联系人工" in normalized:
        return False, "Agent deferred"
    for keyword in ground_truth.get("required", []):
        if keyword.lower() not in normalized:
            return False, f"missing required keyword: {keyword}"
    for keyword in ground_truth.get("forbidden", []):
        if forbidden_keyword_hits(normalized, keyword.lower()):
            return False, f"contains forbidden keyword: {keyword}"
    return True, "correct"


def forbidden_keyword_hits(text: str, keyword: str) -> bool:
    """Match the project's evaluator contract, including short negation prefixes."""
    start = 0
    while True:
        position = text.find(keyword, start)
        if position == -1:
            return False
        prefix = text[max(0, position - 4):position]
        if not any(negation in prefix for negation in ("不", "无", "非", "未", "没")):
            return True
        start = position + 1


def run_llm(version: str, skill: str, question_ids: list[int], model: str) -> dict:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("--llm requires DEEPSEEK_API_KEY")

    data = json.loads((ROOT / "data" / "eval_set.json").read_text(encoding="utf-8"))
    questions = {item["id"]: item for item in data["questions"]}
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    rows = []

    for question_id in question_ids:
        item = questions[question_id]
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE.format(skill=skill)},
                {"role": "user", "content": item["question"]},
            ],
            temperature=0,
            max_tokens=250,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        answer = response.choices[0].message.content.strip()
        correct, reason = evaluate_answer(answer, item["ground_truth"])
        usage = response.usage
        rows.append({
            "id": question_id,
            "question": item["question"],
            "answer": answer,
            "correct": correct,
            "reason": reason,
            "latency_ms": elapsed_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        })

    def total(field: str) -> int | None:
        values = [row[field] for row in rows]
        return sum(values) if all(value is not None for value in values) else None

    return {
        "version": version,
        "model": model,
        "correct": sum(row["correct"] for row in rows),
        "total": len(rows),
        "accuracy": round(sum(row["correct"] for row in rows) / len(rows), 3),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "latency_ms": round(sum(row["latency_ms"] for row in rows), 1),
        "details": rows,
    }


def percent_reduction(before: float, after: float) -> float:
    return round((before - after) / before * 100, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true", help="run the eight-question DeepSeek A/B")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    definition = json.loads(REQUIREMENTS_PATH.read_text(encoding="utf-8"))
    static = {
        name: static_metrics(path, definition["requirements"])
        for name, path in VERSIONS.items()
    }
    before = static["before"]
    after = static["after"]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "shipping_and_delivery Skill optimization",
        "optimization_goal": "preserve policy coverage while reducing prompt tokens and decision ambiguity",
        "static": static,
        "comparison": {
            "character_reduction_percent": percent_reduction(before["characters"], after["characters"]),
            "estimated_token_reduction_percent": percent_reduction(
                before["estimated_tokens"], after["estimated_tokens"]
            ),
            "coverage_change": round(
                after["requirement_coverage"]["rate"] - before["requirement_coverage"]["rate"], 3
            ),
            "information_density_improvement_percent": round(
                (after["covered_requirements_per_1k_tokens"] /
                 before["covered_requirements_per_1k_tokens"] - 1) * 100,
                1,
            ),
        },
        "llm": None,
    }

    if before["requirement_coverage"]["rate"] != 1 or after["requirement_coverage"]["rate"] != 1:
        print("Coverage check failed; optimization dropped required policy facts.", file=sys.stderr)
        for name, metrics in static.items():
            for row in metrics["requirement_coverage"]["details"]:
                if not row["passed"]:
                    print(f"  {name}/{row['id']}: missing {row['missing']}", file=sys.stderr)
        return 1

    if args.llm:
        result["llm"] = {
            name: run_llm(
                name,
                path.read_text(encoding="utf-8"),
                definition["eval_question_ids"],
                args.model,
            )
            for name, path in VERSIONS.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Skill optimization comparison")
    print(f"  Coverage: before {before['requirement_coverage']['rate']:.0%}, "
          f"after {after['requirement_coverage']['rate']:.0%}")
    print(f"  Characters: {before['characters']} -> {after['characters']} "
          f"(-{result['comparison']['character_reduction_percent']}%)")
    print(f"  Estimated tokens: {before['estimated_tokens']} -> {after['estimated_tokens']} "
          f"(-{result['comparison']['estimated_token_reduction_percent']}%)")
    print(f"  Coverage density: {before['covered_requirements_per_1k_tokens']} -> "
          f"{after['covered_requirements_per_1k_tokens']} requirements/1k tokens")
    if result["llm"]:
        for name, llm_result in result["llm"].items():
            print(f"  LLM {name}: {llm_result['correct']}/{llm_result['total']}, "
                  f"prompt_tokens={llm_result['prompt_tokens']}, latency_ms={llm_result['latency_ms']}")
    else:
        print("  LLM A/B: skipped (run with --llm after setting DEEPSEEK_API_KEY)")
    print(f"  Result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
