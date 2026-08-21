"""
Function Calling API 版 ReAct Agent - 支持多轮对话

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

  多轮对话模式
  python react_function_calling_multi_round.py --interactive

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator, List, Dict, Any, Optional

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 回答时要考虑对话历史，结合上下文给出连贯的回答
"""


class Conversation:
    """管理多轮对话的上下文"""

    def __init__(self, system_prompt: str = FC_SYSTEM_PROMPT):
        # 修复：使用正确的变量赋值语法
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.history: List[Dict[str, Any]] = []  # 存储每轮对话的摘要

    def add_user_message(self, question: str) -> None:
        """添加用户信息"""
        self.messages.append({"role": "user", "content": question})
        # 记录历史摘要(不包含工具调用细节，保持简洁)
        self.history.append({
            "role": "user",
            "content": question,
            "timestamp": time.time()
        })

    def add_assistant_message(self, content: str) -> None:
        """添加助手信息"""
        self.messages.append({"role": "assistant", "content": content})
        self.history.append({
            "role": "assistant",
            "content": content[:200] + "..." if len(content) > 200 else content,
            "timestamp": time.time()
        })

    def add_tool_messages(self, tool_calls: List[Any], tool_results: List[Dict]) -> None:
        """添加工具调用和结果"""
        # 只添加工具结果到消息历史，不添加到历史摘要
        for tool_call, result in zip(tool_calls, tool_results):
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result["observation"]
            })

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取当前完整的消息列表"""
        return self.messages

    def get_history_summary(self) -> List[Dict[str, Any]]:
        """获取对话历史摘要(用于显示)"""
        return self.history

    def clear(self):
        """清空对话历史"""
        self.messages = [self.messages[0]]  # 保留 system prompt
        self.history = []


def run(question: str,
        conversation: Optional[Conversation] = None,
        max_steps: int = 10) -> Generator[Dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    if conversation is None:
        conversation = Conversation()

    # 添加用户信息
    conversation.add_user_message(question)
    messages = conversation.get_messages()

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        reason = response.choices[0].finish_reason

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            answer = msg.content or "（模型返回空内容）"
            # 将助手消息加入对话
            conversation.add_assistant_message(answer)
            yield {
                "step": step,
                "type": "final",
                "thought": "",
                "answer": answer,
                "conversation": conversation  # 返回对话对象以便后续使用
            }
            return

        # 模型请求调用工具
        messages.append(msg)

        tool_results = []
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
                "step": step,
                "type": "action",
                "thought": "",  # Function Calling 版 Thought 在模型内部，不可见
                "action": tool_name,
                "action_input": tool_args,
                "observation": str(observation),
            }
            yield step_result

            # 保存工具调用和结果
            tool_results.append({
                "tool_call": tool_call,
                "observation": str(observation),
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(observation),
            })

        # 将工具结果添加到对话中（在循环外统一添加）
        conversation.add_tool_messages(
            [tr["tool_call"] for tr in tool_results],
            [{"observation": tr["observation"]} for tr in tool_results]
        )

    yield {
        "step": max_steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
        "conversation": conversation
    }


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[35m",
    "error": "\033[31m",
    "info": "\033[34m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str,
                  conversation: Optional[Conversation] = None,
                  max_steps: int = 10) -> Optional[Conversation]:
    """运行并打印结果，返回更新后的对话对象"""
    if conversation is None:
        conversation = Conversation()

    # 显示当前对话轮次
    round_num = len([h for h in conversation.get_history_summary() if h["role"] == "user"])

    print(f"\n{'=' * 60}")
    print(f"第 {round_num + 1} 轮对话")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('=' * 60)

    start = time.time()
    final_answer = None

    for step_data in run(question, conversation, max_steps=max_steps):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action", f"🔧 Action:  {step_data['action']}"))
            print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs", f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            final_answer = step_data["answer"]
            print(f"\n{'─' * 60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))
            final_answer = step_data.get("answer", "")

    # 显示对话历史摘要
    print(f"\n{'-' * 60}")
    print(_c("info", "📝 对话历史摘要:"))
    history = conversation.get_history_summary()
    for i, h in enumerate(history):
        role_icon = "👤" if h["role"] == "user" else "🤖"
        content_preview = h["content"][:100] + "..." if len(h["content"]) > 100 else h["content"]
        # 修复：使用正确的字符串拼接方式
        print(f"  {i + 1}. {role_icon} {h['role']}: {content_preview}")

    return conversation


def interactive_mode(max_steps: int = 10):
    """交互式多轮对话模式"""
    print("\n" + "=" * 60)
    print("🤖 A股金融分析助手 - 多轮对话模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("=" * 60 + "\n")

    conversation = Conversation()
    round_num = 0

    while True:
        try:
            # 修复：使用正确的字符串拼接
            question = input(f'\n{_c("info", "👤 你: ")}').strip()
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print(_c('info', '\n👋 再见！'))
                break
            if question.lower() == 'clear':
                conversation.clear()
                print(_c('info', '🗑️  对话历史已清空'))
                continue
            # 执行一轮对话
            conversation = run_and_print(question, conversation, max_steps)
            round_num += 1
        except KeyboardInterrupt:
            print(_c('info', '\n\n👋 再见！'))
            break
        except Exception as e:
            print(_c('error', f'\n❌ 错误: {e}'))
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10, help="最大推理步数")
    parser.add_argument("--interactive", action='store_true', help="进入交互模式")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.max_steps)
    else:
        # 修复：使用关键字参数
        run_and_print(question=args.question, max_steps=args.max_steps)