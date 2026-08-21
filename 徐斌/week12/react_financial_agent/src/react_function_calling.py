"""
Function Calling API 版 ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── LLM 客户端（与手写版一致：默认 DashScope；可用环境变量切换）──────────────
# 文档要求配置 DASHSCOPE_API_KEY。若要用 DeepSeek，可设：
#   export LLM_PROVIDER=deepseek && export DEEPSEEK_API_KEY=sk-xxx
_PROVIDER = os.getenv("LLM_PROVIDER", "dashscope").lower()

if _PROVIDER == "deepseek":
    _api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not _api_key:
        raise SystemExit("错误：LLM_PROVIDER=deepseek 但未设置 DEEPSEEK_API_KEY")
    client = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com")
    MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")
else:
    _api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not _api_key:
        raise SystemExit("错误：未设置 DASHSCOPE_API_KEY（Function Calling 版默认走 DashScope）")
    client = OpenAI(
        api_key=_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    MODEL = os.getenv("AGENT_MODEL", "qwen-max")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


def run(
    question: str,
    max_steps: int = 10,
    history: list[dict] | None = None,
) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    Args:
        question: 本轮用户问题
        max_steps: 单轮最大工具步数
        history: 多轮对话历史（仅含既往 user/assistant 最终问答）

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

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

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            return

        # 模型请求调用工具
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
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
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

    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


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


def run_and_print(
    question: str,
    max_steps: int = 10,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    打印本轮 ReAct 过程，并返回 (最终回答, 更新后的多轮历史)。
    """
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    if history:
        print(f"多轮上下文: 已有 {len(history) // 2} 轮历史")
    print('='*60)

    start = time.time()
    answer = ""

    for step_data in run(question, max_steps=max_steps, history=history):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            answer = step_data["answer"]
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{answer}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            answer = step_data.get("answer", "")
            print(_c("error", f"\n⚠️  {answer}"))

    new_history = list(history or [])
    new_history.append({"role": "user", "content": question})
    new_history.append({"role": "assistant", "content": answer or "（无有效回答）"})
    return answer, new_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)
