"""
compare.py — 三方式对比运行器（教学 centerpiece）

对同一组天气问题，依次跑 Function Call / MCP / CLI(named) / CLI(bash) 四种方式，
记录：调用了哪些工具、耗时、最终答案摘要。
打印对比表 + 写 output/compare_result.md。

使用方式：
  python compare.py
  python compare.py --questions "宁德天气如何？" "北京天气怎么样？"
  python compare.py --provider dashscope

环境变量：DEEPSEEK_API_KEY（默认 LLM）/ DASHSCOPE_API_KEY（备选 LLM）
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
PY = sys.executable

MODES = [
    ("Function Call", [PY, str(BASE_DIR / "mode_function_call" / "run_function_call.py"), "--json", "--quiet"]),
    ("MCP",           [PY, str(BASE_DIR / "mode_mcp" / "run_mcp.py"), "--json", "--quiet"]),
    ("CLI(named)",    [PY, str(BASE_DIR / "mode_cli" / "run_cli.py"), "--mode", "named", "--json", "--quiet"]),
    ("CLI(bash)",     [PY, str(BASE_DIR / "mode_cli" / "run_cli.py"), "--mode", "bash", "--json", "--quiet"]),
]

DEFAULT_QUESTIONS = [
    "宁德现在天气怎么样？",
    "北京今天天气如何？另外未来三天呢？",
    "对比一下上海和深圳现在的天气。",
]


def run_one(mode_cmd: list, question: str, provider: str) -> dict:
    cmd = mode_cmd + ["--provider", provider, "-q", question]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "超时(>180s)", "elapsed": time.time() - t0}

    wall = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[-500:], "elapsed": wall}

    out = proc.stdout.strip().splitlines()
    if not out:
        return {"ok": False, "error": "无输出：" + proc.stderr[-300:], "elapsed": wall}
    try:
        data = json.loads(out[-1])
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"JSON 解析失败：{e}", "elapsed": wall}

    data["ok"] = True
    data["wall_elapsed"] = wall
    return data


def summarize(data: dict) -> dict:
    if not data.get("ok"):
        return {
            "tools": "-",
            "tool_count": 0,
            "llm_elapsed": "-",
            "answer_preview": "(失败) " + (data.get("error", "")[:60]),
        }
    tcs = data.get("tool_calls", [])
    tool_names = ", ".join(t["name"] for t in tcs) or "(无工具调用)"
    answer = data.get("answer", "")
    return {
        "tools": tool_names,
        "tool_count": len(tcs),
        "llm_elapsed": f"{data.get('elapsed', 0):.1f}s",
        "answer_preview": answer[:80].replace("\n", " ") + ("..." if len(answer) > 80 else ""),
    }


def run_compare(questions: list[str], provider: str) -> list[dict]:
    rows = []
    for qi, q in enumerate(questions, 1):
        print(f"\n{'='*70}\nQ{qi}：{q}\n{'='*70}")
        for mode_name, mode_cmd in MODES:
            print(f"  ▶ {mode_name} ...", end=" ", flush=True)
            data = run_one(mode_cmd, q, provider)
            s = summarize(data)
            status = "✓" if data.get("ok") else "✗"
            print(f"{status} 工具[{s['tool_count']}] {s['llm_elapsed']}")
            rows.append({
                "question": q, "mode": mode_name,
                **s, "raw_ok": data.get("ok", False),
            })
    return rows


def write_markdown(rows: list[dict], questions: list[str], provider: str, path: Path):
    lines = [
        "# 三方式对比结果（Function Call / MCP / CLI）",
        "",
        f"- LLM provider：`{provider}`",
        f"- 生成时间：本表由 `python compare.py` 实跑生成",
        f"- 问题数：{len(questions)}，方式数：{len(MODES)}",
        f"- 业务场景：天气查询（geocode → weather-by-coords 多轮循环）",
        "",
        "## 对比表",
        "",
        "| 问题 | 方式 | 工具调用 | 工具数 | LLM耗时 | 答案摘要 |",
        "|------|------|---------|:------:|:-------:|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['question']} | {r['mode']} | {r['tools']} | {r['tool_count']} | "
            f"{r['llm_elapsed']} | {r['answer_preview']} |"
        )

    lines += [
        "",
        "## 解读",
        "",
        "- **工具调用一致性**：四种方式对同一问题通常都走 geocode → weather_by_coords 两步。",
        "- **接入成本**：Function Call 要手写 schema；MCP 要写 Server 但工具自动发现可跨产品复用；"
        "CLI(named) 写白名单；CLI(bash) 几乎零封装但需沙箱。",
        "- **安全**：Function Call / MCP / CLI(named) 都走白名单；CLI(bash) 依赖沙箱拦截。",
        "- **跨模型复用**：MCP 工具可被任意支持 MCP 的 Host 复用；CLI 与模型完全无关。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="三方式对比运行器")
    parser.add_argument("--questions", nargs="+", default=DEFAULT_QUESTIONS)
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "dashscope"])
    args = parser.parse_args()

    print(f"[compare] provider={args.provider}, {len(args.questions)} 个问题 × {len(MODES)} 种方式\n")

    rows = run_compare(args.questions, args.provider)

    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "compare_result.md"
    write_markdown(rows, args.questions, args.provider, out_path)

    print(f"\n{'='*70}\n对比表（已写入 {out_path}）\n{'='*70}")
    print(f"{'问题':<28}{'方式':<16}{'工具数':<6}{'LLM耗时':<10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['question'][:26]:<28}{r['mode']:<16}{r['tool_count']:<6}{r['llm_elapsed']:<10}")


if __name__ == "__main__":
    main()
