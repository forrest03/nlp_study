"""Paired comparison of base-model and GRPO prediction files."""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Any


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            row_id = record.get("row_id")
            if not row_id:
                raise ValueError(f"Missing row_id at {path}:{line_number}")
            if row_id in records:
                raise ValueError(f"Duplicate row_id {row_id!r} in {path}")
            records[row_id] = record
    return records


def _paired_bootstrap_interval(
    differences: list[int], samples: int, seed: int
) -> tuple[float, float]:
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    rng = random.Random(seed)
    size = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size for _ in range(samples)
    )
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def _mcnemar_exact_p_value(improved: int, regressed: int) -> float:
    """Two-sided exact sign test over discordant pairs."""
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = Fraction(
        sum(__import__("math").comb(discordant, k) for k in range(tail + 1)), 2**discordant
    )
    return min(1.0, 2.0 * float(probability))


def compare(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:5]
        missing_baseline = sorted(set(candidate) - set(baseline))[:5]
        raise ValueError(
            "Prediction files must contain identical row_ids. "
            f"Missing from candidate: {missing_candidate}; missing from baseline: {missing_baseline}"
        )
    if not baseline:
        raise ValueError("Prediction files are empty")

    row_ids = sorted(baseline)
    baseline_scores = [int(bool(baseline[row_id]["correct"])) for row_id in row_ids]
    candidate_scores = [int(bool(candidate[row_id]["correct"])) for row_id in row_ids]
    differences = [new - old for old, new in zip(baseline_scores, candidate_scores)]
    lower, upper = _paired_bootstrap_interval(differences, bootstrap_samples, seed)
    improved = sum(value == 1 for value in differences)
    regressed = sum(value == -1 for value in differences)
    total = len(row_ids)

    return {
        "examples": total,
        "baseline_exact_match": sum(baseline_scores) / total,
        "candidate_exact_match": sum(candidate_scores) / total,
        "absolute_improvement": sum(differences) / total,
        "paired_bootstrap_95pct_ci": [lower, upper],
        "improved_examples": improved,
        "regressed_examples": regressed,
        "unchanged_examples": total - improved - regressed,
        "mcnemar_exact_p_value": _mcnemar_exact_p_value(improved, regressed),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two paired GSM8K evaluations")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="results/comparison.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = compare(
        load_predictions(args.baseline),
        load_predictions(args.candidate),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
