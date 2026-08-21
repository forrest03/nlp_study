"""Parallel vs Serial A/B：凸显 subagent 并行加速。"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from agents import run_cake_research

QUESTIONS = [
    "采集生日蛋糕类商品详情（图片+文字介绍+价格），并给出营销设计方案",
    "芝士蛋糕市场：热销商品采集与下午茶场景营销方案",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2)
    args = ap.parse_args()
    qs = QUESTIONS[: max(1, args.limit)]
    rows = []
    for q in qs:
        print(f"\n=== {q[:40]}... ===")
        rp = run_cake_research(q, serial=False)
        rs = run_cake_research(q, serial=True)
        ps = (rp["parallel_stats"] or [{}])[-1]
        ss = (rs["parallel_stats"] or [{}])[-1]
        row = {
            "question": q,
            "parallel_wall": ps.get("wall_clock"),
            "serial_wall": ss.get("wall_clock"),
            "dispatch_speedup_parallel": ps.get("speedup"),
            "n_subagents": ps.get("n_subagents") or len(rp["subagents"]),
        }
        rows.append(row)
        print(
            f"  parallel wall={row['parallel_wall']}s | serial wall={row['serial_wall']}s "
            f"| dispatch speedup={row['dispatch_speedup_parallel']}×"
        )

    out = Path(__file__).parent.parent / "outputs" / "eval_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
