"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent（支持会话记忆）

使用方式：
  python agent.py
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"
  python agent.py --mode manual   --question "..." --max_steps 8
  python agent.py --mode manual   --question "..." --session_id "sess001"

环境变量：
  DASHSCOPE_API_KEY  必填
  AGENT_MODEL        默认 qwen-max，可换 deepseek-v3 等

新增功能：
  - 支持 --session_id 参数，按会话管理历史对话
  - 自动检索历史问答并注入上下文
  - 自动保存当前问答到会话历史
"""

import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent（带记忆）")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--session_id", help="会话ID，用于关联历史对话")
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print

    run_and_print(args.question, args.max_steps, args.session_id)