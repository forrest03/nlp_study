"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent（支持多轮对话）

使用方式：
  # 交互多轮（默认）
  python agent.py
  python agent.py --mode fc

  # 指定首问后继续多轮
  python agent.py --mode manual --question "茅台2023年毛利率是多少？"

  # 只跑一轮后退出（脚本/评测用）
  python agent.py --once --question "五粮液近一年股价涨跌幅？"

环境变量：
  DASHSCOPE_API_KEY  必填（manual / fc 默认均走 DashScope）
  AGENT_MODEL        默认 qwen-max
  LLM_PROVIDER       可选 deepseek（需同时设 DEEPSEEK_API_KEY）
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

QUIT_CMDS = {"q", "quit", "exit", "退出"}
CLEAR_CMDS = {"clear", "/clear", "清空"}


def chat_loop(run_and_print, max_steps: int, first_question: str | None = None, once: bool = False):
    """
    多轮对话循环：每轮结束后把 user 问题 + Final Answer 写入 history，
    下一轮作为上下文传入，使 Agent 能理解指代（如「那另一家呢？」）。
    """
    history: list[dict] = []
    turn = 0

    print("\n进入多轮对话。输入问题开始；clear 清空历史；quit / exit / q 退出。")
    if once:
        print("（--once：仅执行一轮）")

    pending = first_question
    while True:
        if pending is not None:
            question = pending.strip()
            pending = None
        else:
            try:
                question = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已退出。")
                break

        if not question:
            continue
        if question.lower() in QUIT_CMDS or question in QUIT_CMDS:
            print("已退出。")
            break
        if question.lower() in CLEAR_CMDS or question in CLEAR_CMDS:
            history = []
            turn = 0
            print("已清空多轮历史。")
            continue

        turn += 1
        print(f"\n—— 第 {turn} 轮 ——")
        _, history = run_and_print(question, max_steps=max_steps, history=history)

        if once:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent（多轮对话）")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument(
        "--question", "-q", default=None,
        help="可选：首轮问题；不传则进入交互输入",
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument(
        "--once", action="store_true",
        help="只回答一轮后退出（需配合 --question）",
    )
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print

    if args.once and not args.question:
        parser.error("--once 需要同时提供 --question")

    chat_loop(
        run_and_print,
        max_steps=args.max_steps,
        first_question=args.question,
        once=args.once,
    )
