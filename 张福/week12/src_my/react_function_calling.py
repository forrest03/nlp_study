"""
Function Calling API 版 ReAct Agent（支持会话记忆）

新增功能：
  1. 支持 session_id 参数，按会话管理历史对话
  2. 在发送提问时，自动检索同一会话的历史问答并注入上下文
  3. 问答完成后，自动保存当前问答到会话历史

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台毛利率" --session_id "sess001"
  python react_function_calling.py --question "比五粮液高多少？" --session_id "sess001"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator, Optional

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT_TEMPLATE = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 注意参考历史对话记忆，理解上下文关联

{history_section}
"""

HISTORY_TEMPLATE = """

历史对话记忆（与当前问题相关的过往问答）：
{history_items}
"""


def _build_history_section(history: list) -> str:
    """构建历史对话记忆部分的prompt"""
    if not history:
        return ""

    items = []
    for i, item in enumerate(history, 1):
        items.append(f"[{i}] 用户问：{item['question']}\n   助手答：{item['answer']}")

    return HISTORY_TEMPLATE.format(history_items="\n".join(items))


def run(question: str, max_steps: int = 10,
        session_id: Optional[str] = None, conv_store=None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    Args:
        question: 用户问题
        max_steps: 最大步数
        session_id: 会话ID（可选）
        conv_store: 会话存储实例（可选）

    Returns:
        每一步的结构化结果
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    history = []
    if session_id and conv_store:
        history = conv_store.search(session_id, question, top_k=3)
        logger.info(f"检索到历史对话: {len(history)} 条")

    history_section = _build_history_section(history)
    system_prompt = FC_SYSTEM_PROMPT_TEMPLATE.format(history_section=history_section)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    final_answer = None

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg    = response.choices[0].message
        reason = response.choices[0].finish_reason

        if reason == "stop" or not msg.tool_calls:
            final_answer = msg.content or "（模型返回空内容）"
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": final_answer,
            }
            break

        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(observation),
            })

    else:
        final_answer = f"已达最大步数 {max_steps}，未能得出最终答案"
        yield {
            "step":   max_steps + 1,
            "type":   "max_steps",
            "answer": final_answer,
        }

    if session_id and conv_store and final_answer:
        conv_store.add(session_id, question, final_answer)
        logger.info(f"已保存问答到会话: {session_id}")


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10, session_id: Optional[str] = None):
    """执行并打印结果"""
    from conversation_store import get_conv_store

    conv_store = get_conv_store()

    print(f"\n{'='*60}")
    print(f"问题: {question}")
    if session_id:
        print(f"会话: {session_id}")
    print(f"模型: {MODEL}  实现: Function Calling（带记忆）")
    print('='*60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps, session_id=session_id, conv_store=conv_store):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--session_id", help="会话ID，用于关联历史对话")
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps, args.session_id)