"""
Harness 工程统一入口

使用方式：
  python harness/agent.py                                           # 交互式对话
  python harness/agent.py --question "你好，记得我的偏好吗？"          # 单次提问
  python harness/agent.py --allow-tools                             # 交互式对话（启用工具调用）
  python harness/agent.py --allow-tools --question "创建 Python 项目"
  python harness/agent.py --compress                                # 手动压缩长期记忆
  python harness/agent.py --init-index                              # 从 databases/memory_meta.json 重建向量索引

工具调用（--allow-tools）：
  LLM 可通过 function call 自动创建目录、读写文件、执行 shell 命令。
  执行 shell 命令前会请求用户确认。

环境变量：
  DASHSCOPE_API_KEY  必填
  HARNESS_MODEL      可选，默认 qwen-plus

目录说明：
  memory/      Markdown 记忆文件（短期/长期/用户特征/按日）
  databases/   BM25 + RAG 检索索引（memory_meta.json + FAISS）
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    parser = argparse.ArgumentParser(description="Harness LLM 对话（带记忆系统）")
    parser.add_argument("--question", type=str, default=None, help="单次提问")
    parser.add_argument("--session_id", type=str, default=None, help="会话 ID")
    parser.add_argument("--model", type=str, default=os.getenv("HARNESS_MODEL", "qwen-plus"))
    parser.add_argument("--compress", action="store_true", help="手动触发长期记忆压缩")
    parser.add_argument("--init-index", action="store_true", help="从 compressed 重建向量索引")
    parser.add_argument(
        "--allow-tools",
        action="store_true",
        help="启用 function call 工具（创建目录、写文件、执行命令）",
    )
    args = parser.parse_args()

    from harness.chat_harness import (
        run_chat,
        interactive_loop,
        init_demo_vectorstore,
    )
    from src.memory_compressor import compress_raw_memories, rebuild_vector_index
    from src.memory_retriever import reload_retriever

    init_demo_vectorstore()

    if args.compress:
        compress_raw_memories()
        reload_retriever()
        print("长期记忆压缩完成。")
        return

    if args.init_index:
        rebuild_vector_index()
        reload_retriever()
        print("向量索引重建完成。")
        return

    if args.question:
        print(f"会话: {args.session_id or '(新建)'}")
        print(f"问题: {args.question}\n")
        print("工具调用: {'已启用' if args.allow_tools else '已禁用'}")
        print("助手: ", end="", flush=True)
        answer, sid = run_chat(
            args.question,
            session_id=args.session_id,
            stream=not args.allow_tools,
            model=args.model,
            allow_tools=args.allow_tools,
        )
        print(f"\n[session_id={sid}]")
    else:
        interactive_loop(
            session_id=args.session_id,
            model=args.model,
            allow_tools=args.allow_tools,
        )


if __name__ == "__main__":
    main()
