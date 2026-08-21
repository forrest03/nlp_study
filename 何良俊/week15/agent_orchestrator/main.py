"""
main — CLI 入口
================

用法：
  # 内置 demo：大模型推理优化综述（性能 / 成本 / 生态 三个子主题并行调研）
  python main.py --demo

  # 自定义任意可拆分的复杂任务
  python main.py --question "请从 X / Y / Z 三个角度调研……并汇总成报告"

  # 调整并行度 / 步数上限 / 保存完整报告
  python main.py --question "……" --max-workers 4 --save-json report.json

配置：
  set DEEPSEEK_API_KEY=sk-xxx          # 必填
  set DEEPSEEK_MODEL=deepseek-v4-flash # 可选
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from display import on_event, print_banner, print_usage  # noqa: E402
from llm import DeepSeekClient                            # noqa: E402
from orchestrator import OrchestratorAgent                # noqa: E402

DEMO_QUESTION = (
    "我想系统了解大模型推理优化。请分别从「推理性能优化技术」「推理成本优化策略」"
    "「主流推理框架与生态」三个子主题进行并行调研，每个子主题输出结构化要点，"
    "最后汇总成一份完整的综述报告。"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="可并行下发 subagent 的编排 Agent")
    p.add_argument("--question", default=None, help="要编排的复杂任务（缺省用内置 demo 问题）")
    p.add_argument("--demo", action="store_true", help="使用内置 demo 问题")
    p.add_argument("--max-workers", type=int, default=3, help="subagent 并行度（默认 3）")
    p.add_argument("--max-steps", type=int, default=12, help="编排者决策循环步数上限")
    p.add_argument("--subagent-max-steps", type=int, default=6, help="每个 subagent 的步数上限")
    p.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"), help="模型名")
    p.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"), help="API 地址")
    p.add_argument("--save-json", default=None, help="把完整报告（含 subagent 结果）另存为 JSON 文件")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    question = args.question or DEMO_QUESTION

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 未配置 DEEPSEEK_API_KEY 环境变量，无法调用 LLM。", file=sys.stderr)
        print("  PowerShell: $env:DEEPSEEK_API_KEY = \"sk-xxx\"", file=sys.stderr)
        print("  CMD:        set DEEPSEEK_API_KEY=sk-xxx", file=sys.stderr)
        return 1

    client = DeepSeekClient(api_key=api_key, base_url=args.base_url, model=args.model)
    orchestrator = OrchestratorAgent(
        client,
        max_workers=args.max_workers,
        subagent_max_steps=args.subagent_max_steps,
        on_event=on_event,
    )

    print_banner(question, args.model, args.max_workers)

    try:
        result = orchestrator.run(question)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except RuntimeError as e:
        print(f"\n错误: {e}", file=sys.stderr)
        return 2

    print_usage(client.usage)

    if args.save_json:
        report = {
            "question": question,
            "model": args.model,
            "orchestrator_result": result,
            "subagent_results": orchestrator.last_dispatch,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"💾 完整报告已保存: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
