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

# ── LLM 客户端（懒加载，避免导入时因缺少环境变量而报错）─────────────────────────
_client = None
MODEL = os.getenv("AGENT_MODEL", "deepseek-chat")

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _client

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


def _serialize_messages(messages, max_content=400):
    """
    把 messages 列表序列化为可 JSON 化的纯 dict，供前端可视化展示。
    messages 中可能混杂 dict 和 OpenAI 返回的 message 对象，需分别处理。
    """
    result = []
    for m in messages:
        if hasattr(m, "role"):  # OpenAI message 对象
            item = {"role": m.role, "content": (m.content or "")}
            if getattr(m, "tool_calls", None):
                item["tool_calls"] = [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in m.tool_calls
                ]
        else:  # dict
            item = {"role": m.get("role", "?"), "content": str(m.get("content", "") or "")}
            if m.get("tool_calls"):
                item["tool_calls"] = [
                    {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                    for tc in m["tool_calls"]
                ]
            if m.get("tool_call_id"):
                item["tool_call_id"] = m["tool_call_id"]
        # 截断过长内容，避免 SSE 负载过大
        if len(item["content"]) > max_content:
            item["content"] = item["content"][:max_content] + f" …(截断，原文共{len(item['content'])}字符)"
        result.append(item)
    return result


def run(question: str, max_steps: int = 10, messages: list = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    多轮对话：
      传入 messages 可延续上一轮对话历史，函数会就地修改该列表。
      首次调用不传 messages，从最终 yield 的 "messages" 字段获取，后续传入即可。

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    if messages is None:
        messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": question})

    call_index = 0
    for step in range(1, max_steps + 1):
        call_index += 1
        # 调用 LLM 前先推送一份 prompt 快照，供前端可视化"提示词是如何拼接的"
        yield {
            "type":       "llm_call",
            "call_index": call_index,
            "step":       step,
            "messages":   _serialize_messages(messages),
        }

        response = _get_client().chat.completions.create(
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
            messages.append({"role": "assistant", "content": msg.content or ""})
            yield {
                "step":     step,
                "type":     "final",
                "thought":  "",
                "answer":   msg.content or "（模型返回空内容）",
                "messages": messages,
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
        "step":     max_steps + 1,
        "type":     "max_steps",
        "answer":   f"已达最大步数 {max_steps}，未能得出最终答案",
        "messages": messages,
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


def run_and_print(question: str, max_steps: int = 10):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
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


def run_interactive(max_steps: int = 10):
    """多轮对话交互模式"""
    print(f"\n{'='*60}")
    print(f"ReAct 金融分析助手 - 多轮对话模式")
    print(f"模型: {MODEL}  实现: Function Calling")
    print(f"输入 quit/exit 退出 | reset 重置对话 | /history 查看历史")
    print('='*60)

    messages = None  # 对话历史，首次调用后自动初始化
    turn = 0

    while True:
        try:
            question = input(f"\n📝 [第{turn + 1}轮] 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if question.lower() == "reset":
            messages = None
            turn = 0
            print("✅ 对话已重置。")
            continue
        if question.lower() == "/history":
            if messages is None:
                print("（暂无对话历史）")
            else:
                for m in messages:
                    role = m.get("role", "?")
                    content = m.get("content", "")
                    if isinstance(content, str) and content:
                        print(f"  [{role}] {content[:120]}")
            continue

        turn += 1
        start = time.time()

        for step_data in run(question, max_steps=max_steps, messages=messages):
            stype = step_data["type"]

            if stype == "action":
                print(f"\n  [Step {step_data['step']}]")
                print(_c("action",  f"  🔧 {step_data['action']}({json.dumps(step_data['action_input'], ensure_ascii=False)})"))
                print(_c("obs",     f"  👁  {step_data['observation'][:200]}"))

            elif stype == "final":
                messages = step_data["messages"]
                elapsed = time.time() - start
                print(f"\n{'─'*60}")
                print(_c("final", f"✅ {step_data['answer']}"))
                print(f"  （{step_data['step']} 步，{elapsed:.1f}s）")

            elif stype in ("error", "max_steps"):
                messages = step_data.get("messages", messages)
                print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default=None, help="单轮问题，不指定则进入多轮交互模式")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--interactive", action="store_true", help="强制进入多轮交互模式")
    args = parser.parse_args()

    if args.interactive or args.question is None:
        run_interactive(args.max_steps)
    else:
        run_and_print(args.question, args.max_steps)
