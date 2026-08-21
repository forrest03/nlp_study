"""Parallel vs Serial 量化对比（凸显 subagent 并行优势，股票版）

使用方式：
  python eval_compare.py            # 默认 2 题，parallel vs serial
  python eval_compare.py --limit 1  # 快速版
"""
import sys, time, json, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_CASES = [
    ("比亚迪", "2026-08-04"),
    ("宁德时代", "2026-08-05"),
]


def run_one(company, date_str, serial):
    """跑一次股票多空分析，返回统计。serial 控制子 agent 执行方式。"""
    import agents
    question = f"查询 {company} 在 {date_str} 的股票，给出多空分析"
    t0 = time.time()
    r = agents.run_research(question, serial=serial)
    wall = time.time() - t0
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else None
    return {
        "wall": round(wall, 2),
        "n_subagents": ps["n_subagents"] if ps else 0,
        "dispatch_wall": ps["wall_clock"] if ps else 0,
        "serial_sum": ps["serial_sum"] if ps else 0,
        "speedup": ps["speedup"] if ps else 0,
        "dispatched": len(r["dispatches"]) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="parallel vs serial 对比（股票）")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cases = EVAL_CASES[:args.limit] if args.limit else EVAL_CASES

    results = []
    for i, (company, date_str) in enumerate(cases):
        logger.warning(f"[{i+1}/{len(cases)}] {company} {date_str}")
        p = run_one(company, date_str, serial=False)
        s = run_one(company, date_str, serial=True)
        results.append({"company": company, "date": date_str,
                        "parallel": p, "serial": s})
        print(f"  {company} {date_str:<12} 并行 {p['wall']}s vs 串行 {s['wall']}s "
              f"(subagent {p['n_subagents']}, 加速 {p['speedup']}×)")

    if not results:
        print("无结果")
        return

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)

    print(f"\n{'='*60}\nParallel vs Serial 对比（{len(results)} 题）\n{'='*60}")
    print(f"{'指标':<16} {'并行(ThreadPool)':<18} {'串行(for循环)':<18}")
    print(f"{'平均墙钟(s)':<16} {avg_p:<18.2f} {avg_s:<18.2f}")
    print(f"{'平均加速':<16} {avg_spd:<18.2f}× {'—':<18}")
    print(f"\n结论：看多/看空 subagent 并行把 2 个独立分析任务的墙钟从 sum 压到 ≈max，"
          f"平均加速 {avg_spd:.2f}×")

    out = {"summary": {"avg_parallel_s": round(avg_p, 2),
                       "avg_serial_s": round(avg_s, 2),
                       "avg_speedup": round(avg_spd, 2)},
           "details": results}
    out_path = Path(__file__).parent.parent / "outputs" / "eval_compare.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
