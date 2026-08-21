"""
手写 Prompt 解析版 ReAct Agent（支持会话记忆）

新增功能：
  1. 支持 session_id 参数，按会话管理历史对话
  2. 在发送提问时，自动检索同一会话的历史问答并注入上下文
  3. 问答完成后，自动保存当前问答到会话历史

使用方式：
  python react_manual.py
  python react_manual.py --question "茅台毛利率" --session_id "sess001"
  python react_manual.py --question "比五粮液高多少？" --session_id "sess001"
"""

import os
import re
import json
import time
import logging
import argparse
from typing import Generator, Optional

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """你是一个专业的A股金融分析助手，可以使用以下工具来回答问题：

工具列表：
1. rag_search(query) - 在年报中语义检索文本内容（战略/财务数据/风险因素等）
2. company_lookup(name) - 将公司名称转换为股票代码
3. calculator(expr) - 计算数学表达式（支持四则运算和math函数）
4. financial_indicator(symbol) - 获取实时财务指标（PE/PB/ROE等）
5. stock_price(symbol, start_date, end_date) - 获取历史股价，日期格式YYYYMMDD

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {{"参数名": "参数值"}}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 必须先用 company_lookup 获取股票代码，再调用 financial_indicator 或 stock_price
- 数字计算必须用 calculator，不能心算
- Final Answer 必须引用具体数据来源（哪份年报哪一页，或AkShare实时数据）
- 如果没有合适工具能回答，直接输出 Final Answer 说明原因
- 注意参考历史对话记忆，理解上下文关联

{history_section}
"""

HISTORY_TEMPLATE = """

历史对话记忆（与当前问题相关的过往问答）：
{history_items}
"""


# ── 格式解析 ──────────────────────────────────────────────────────────────────
_THOUGHT_RE      = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE       = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE        = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse_step(text: str) -> dict:
    """从 LLM 输出中解析一步的结构化内容"""
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type":    "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer":  final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m  = _ACTION_RE.search(text)
    input_m   = _ACTION_INPUT_RE.search(text)

    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type":         "action",
        "thought":      thought_m.group(1).strip() if thought_m else "",
        "action":       action_m.group(1).strip(),
        "action_input": action_input,
    }


def _build_history_section(history: list) -> str:
    """构建历史对话记忆部分的prompt"""
    if not history:
        return ""

    items = []
    for i, item in enumerate(history, 1):
        items.append(f"[{i}] 用户问：{item['question']}\n   助手答：{item['answer']}")

    return HISTORY_TEMPLATE.format(history_items="\n".join(items))


# ── ReAct 核心循环 ─────────────────────────────────────────────────────────────

def run(question: str, max_steps: int = 10, verbose: bool = True,
        session_id: Optional[str] = None, conv_store=None) -> Generator[dict, None, None]:
    """
    执行 ReAct 循环，yield 每一步的结构化结果

    Args:
        question: 用户问题
        max_steps: 最大步数
        verbose: 是否打印详细信息
        session_id: 会话ID（可选）
        conv_store: 会话存储实例（可选）

    Returns:
        每一步的结构化结果
    """
    from tools import TOOLS_MAP

    # 获取历史对话（如果有会话ID）
    history = []
    if session_id and conv_store:
        history = conv_store.search(session_id, question, top_k=3)
        logger.info(f"检索到历史对话: {len(history)} 条")

    # 构建包含历史记忆的system prompt
    history_section = _build_history_section(history)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(history_section=history_section)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]

    final_answer = None

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],
        )
        llm_output = response.choices[0].message.content.strip()
        parsed = _parse_step(llm_output)

        if parsed["type"] == "final":
            final_answer = parsed["answer"]
            yield {
                "step":    step,
                "type":    "final",
                "thought": parsed["thought"],
                "answer":  final_answer,
            }
            break

        if parsed["type"] == "unparseable":
            yield {
                "step":        step,
                "type":        "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
            }
            return

        tool_name  = parsed["action"]
        tool_args  = parsed["action_input"]
        tool_fn    = TOOLS_MAP.get(tool_name)

        if tool_fn is None:
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
        else:
            try:
                observation = tool_fn(**tool_args)
            except TypeError as e:
                observation = f"工具参数错误: {e}"

        step_result = {
            "step":         step,
            "type":         "action",
            "thought":      parsed["thought"],
            "action":       tool_name,
            "action_input": tool_args,
            "observation":  str(observation),
        }
        yield step_result

        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role":    "user",
            "content": f"Observation: {observation}\n",
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


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {
    "thought":  "\033[36m",
    "action":   "\033[33m",
    "obs":      "\033[32m",
    "final":    "\033[35m",
    "error":    "\033[31m",
    "reset":    "\033[0m",
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
    print(f"模型: {MODEL}  实现: 手写Prompt解析（带记忆）")
    print('='*60)

    start = time.time()
    step_count = 0

    for step_data in run(question, max_steps=max_steps, session_id=session_id, conv_store=conv_store):
        step_count += 1
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            if step_data.get("thought"):
                print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("final",  f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--session_id", help="会话ID，用于关联历史对话")
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps, args.session_id)