"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  python agent.py                          # 默认进入多轮对话交互模式
  python agent.py --mode fc                # Function Calling 版多轮对话
  python agent.py --question "茅台毛利率？"  # 单轮模式
  python agent.py --mode manual --question "..." --max_steps 8

环境变量：
  DEEPSEEK_API_KEY  必填
  AGENT_MODEL       默认 deepseek-chat
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="fc",
        help="manual=手写Prompt解析版  fc=Function Calling版（默认）",
    )
    parser.add_argument("--question", default=None, help="指定则单轮执行，不指定则进入多轮交互")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run_and_print, run_interactive
    else:
        from react_function_calling import run_and_print, run_interactive

    if args.question:
        run_and_print(args.question, args.max_steps)
    else:
        run_interactive(args.max_steps)
