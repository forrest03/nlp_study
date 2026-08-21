"""
Function Calling API 版 ReAct Agent —— 多轮对话增强版

教学重点：
  1. 在 react_function_calling.py 基础上引入 Session 抽象，承载跨轮次的短期记忆
  2. 每轮对话复用历史 messages，模型可以引用“刚才查到的茅台毛利率”等上下文
  3. 滑动窗口 + 可选摘要压缩，防止长对话 token 爆炸
  4. 提供 REPL 交互模式，连续提问、追问、澄清

使用方式：
  # 单轮（兼容原版用法，但不带记忆）
  python react_function_calling_multiturn.py --question "茅台2023年毛利率是多少？"

  # 多轮交互 REPL（推荐，演示短期记忆）
  python react_function_calling_multiturn.py --interactive

  # 持久化会话：退出后下次可继续
  python react_function_calling_multiturn.py --interactive --session-file session.json
  python react_function_calling_multiturn.py --interactive --session-file session.json --resume

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DEEPSEEK_API_KEY="sk-xxx"
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

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 你正在与用户进行多轮对话，可以引用之前轮次中已查到的数据，避免重复调用相同工具
- 如果用户的问题依赖前文（如"那它近一年股价呢？"中的"它"），请结合历史对话推断指代对象
"""


# ── Session：短期记忆容器 ─────────────────────────────────────────────────────

class ConversationSession:
    """
    跨轮次对话的短期记忆容器。

    - messages: 完整 OpenAI messages 列表（含 system / user / assistant / tool）
    - max_history_pairs: 滑动窗口，保留最近 N 轮 user+assistant 对话，超出的截断
    - 自动持久化到 JSON 文件（可选）
    """

    def __init__(
        self,
        system_prompt: str = FC_SYSTEM_PROMPT,
        max_history_pairs: int = 6,
        session_file: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.max_history_pairs = max_history_pairs
        self.session_file = session_file
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # 已完成的对话轮次（不含当前进行中的），用于滑动窗口计数
        self.completed_turns: list[dict] = []  # 每项: {"user":..., "assistant":...}
        if session_file and os.path.exists(session_file):
            self.load()

    # —— 记忆管理 ——

    def add_user_message(self, content: str) -> None:
        """开始新一轮对话：追加 user 消息"""
        self.messages.append({"role": "user", "content": content})

    def finalize_turn(self, assistant_msg: dict) -> None:
        """
        一轮对话结束（拿到 final answer）时调用。
        assistant_msg 为最终 assistant 消息（不含 tool_calls 的纯文本回复）。
        """
        self.messages.append(assistant_msg)
        user_msg = self._find_last_user_msg()
        self.completed_turns.append({
            "user": user_msg,
            "assistant": assistant_msg.get("content", ""),
        })
        self._enforce_window()
        self.save()

    def _find_last_user_msg(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def _enforce_window(self) -> None:
        """
        滑动窗口：当历史轮次超过 max_history_pairs 时，
        丢弃最早的一轮（包括其产生的 user / assistant / tool 全部消息），
        避免长对话 token 爆炸。
        """
        if len(self.completed_turns) <= self.max_history_pairs:
            return
        drop_count = len(self.completed_turns) - self.max_history_pairs
        # 找到第一条非 system 消息作为截断起点
        non_system_start = 0
        for i, m in enumerate(self.messages):
            if m.get("role") != "system":
                non_system_start = i
                break
        # 从 non_system_start 开始，移除前 drop_count 轮的所有消息
        removed_turns = 0
        removed_idx = non_system_start
        while removed_turns < drop_count and removed_idx < len(self.messages):
            msg = self.messages[removed_idx]
            # 一轮 = user + (assistant tool_calls? + tool*) + assistant final
            if msg.get("role") == "user":
                removed_turns += 1
            removed_idx += 1
        # 截断
        self.messages = self.messages[:non_system_start] + self.messages[removed_idx:]
        self.completed_turns = self.completed_turns[drop_count:]
        logger.info(
            f"[memory] 滑动窗口触发，丢弃 {removed_turns} 轮早期对话，"
            f"当前消息数 {len(self.messages)}"
        )

    # —— 持久化 ——

    def save(self) -> None:
        if not self.session_file:
            return
        try:
            data = {
                "system_prompt": self.system_prompt,
                "messages": self.messages,
                "completed_turns": self.completed_turns,
                "max_history_pairs": self.max_history_pairs,
            }
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[memory] 保存会话失败: {e}")

    def load(self) -> None:
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.system_prompt = data.get("system_prompt", self.system_prompt)
            self.messages = data.get("messages", self.messages)
            self.completed_turns = data.get("completed_turns", [])
            logger.info(
                f"[memory] 已加载会话 {self.session_file}，"
                f"消息数 {len(self.messages)}，历史轮次 {len(self.completed_turns)}"
            )
        except Exception as e:
            logger.warning(f"[memory] 加载会话失败: {e}")

    def reset(self) -> None:
        """清空对话历史，保留 system prompt"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.completed_turns = []
        self.save()

    def brief(self) -> str:
        """返回简短的会话概要，用于 REPL 显示"""
        if not self.completed_turns:
            return "（无历史）"
        lines = []
        for i, t in enumerate(self.completed_turns[-3:], 1):
            u = t["user"][:40].replace("\n", " ")
            a = (t["assistant"] or "")[:40].replace("\n", " ")
            lines.append(f"  {i}. Q: {u}... → A: {a}...")
        return "\n".join(lines)


# ── 单轮 ReAct 循环（基于 session 的 messages 续写） ──────────────────────────

def run(
    question: str,
    session: ConversationSession,
    max_steps: int = 10,
) -> Generator[dict, None, None]:
    """
    在指定 session 上执行一轮 Function Calling ReAct 循环。

    与原版 react_function_calling.run() 的区别：
      - 不再每次新建 messages，而是从 session.history 续写
      - 循环结束后调用 session.finalize_turn()，把最终答案写回记忆
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    session.add_user_message(question)
    messages = session.messages

    final_answer: Optional[str] = None

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

        # 将 ChatCompletionMessage 对象转换为字典
        msg_dict = {
            "role":    msg.role,
            "content": msg.content or "",
        }
        if msg.tool_calls:
            msg_dict["tool_calls"] = []
            for tc in msg.tool_calls:
                msg_dict["tool_calls"].append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            final_answer = msg.content or "（模型返回空内容）"
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": final_answer,
            }
            break

        # 模型请求调用工具
        messages.append(msg_dict)

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
    else:
        final_answer = f"已达最大步数 {max_steps}，未能得出最终答案"
        yield {
            "step":   max_steps + 1,
            "type":   "max_steps",
            "answer": final_answer,
        }

    # 把最终答案固化为一条 assistant 消息，写入短期记忆
    # 注意：tool_calls 链路上的中间 assistant 消息已 append，
    # 这里追加的是"最终结论"消息，让下一轮对话可以引用
    if final_answer is not None:
        session.finalize_turn({"role": "assistant", "content": final_answer})


# ── CLI 打印（复用原版彩色输出） ───────────────────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "memory":  "\033[34m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, session: ConversationSession, max_steps: int = 10):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling (多轮记忆)")
    if session.completed_turns:
        print(_c("memory", f"📌 记忆上下文: 共 {len(session.completed_turns)} 轮历史"))
        print(_c("memory", session.brief()))
    print('='*60)

    start = time.time()

    for step_data in run(question, session, max_steps=max_steps):
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
            print(f"\n本轮 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


# ── REPL 交互模式 ─────────────────────────────────────────────────────────────

HELP_TEXT = """
可用命令：
  /help          显示此帮助
  /history       查看当前会话历史摘要
  /reset         清空会话记忆，开始新对话
  /save [path]   保存会话到文件（默认使用 --session-file）
  /exit 或 /quit 退出
直接输入问题即可继续对话（可引用上文，如"那它近一年股价呢？"）
""".strip()


def interactive_repl(session: ConversationSession, max_steps: int = 10):
    print(f"\n{'#'*60}")
    print("#  多轮对话 ReAct Agent (Function Calling 版)")
    print(f"#  模型: {MODEL}")
    print(f"#  滑动窗口: 最近 {session.max_history_pairs} 轮")
    if session.session_file:
        print(f"#  会话文件: {session.session_file}")
    print('#  输入 /help 查看命令，/exit 退出')
    print(f"{'#'*60}")

    if session.completed_turns:
        print(_c("memory", f"\n📌 已恢复 {len(session.completed_turns)} 轮历史对话："))
        print(_c("memory", session.brief()))

    while True:
        try:
            question = input("\n💬 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue

        # 命令处理
        if question.startswith("/"):
            cmd, *rest = question[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd in ("exit", "quit"):
                print("再见！")
                break
            elif cmd == "help":
                print(HELP_TEXT)
            elif cmd == "history":
                print(_c("memory", f"当前历史轮次: {len(session.completed_turns)}"))
                print(_c("memory", session.brief()))
            elif cmd == "reset":
                session.reset()
                print(_c("memory", "🔄 会话已清空，开始新对话"))
            elif cmd == "save":
                path = arg or session.session_file
                if path:
                    session.session_file = path
                    session.save()
                    print(_c("memory", f"💾 已保存到 {path}"))
                else:
                    print("用法: /save <path>")
            else:
                print(f"未知命令: /{cmd}，输入 /help 查看可用命令")
            continue

        # 普通对话
        try:
            run_and_print(question, session, max_steps=max_steps)
        except Exception as e:
            print(_c("error", f"\n⚠️  执行出错: {e}"))


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轮对话 Function Calling ReAct Agent")
    parser.add_argument("--question",  default=None, help="单轮问题（不指定则进入 REPL）")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--interactive", action="store_true", help="进入多轮交互 REPL")
    parser.add_argument("--session-file", default=None, help="会话持久化文件路径")
    parser.add_argument("--resume", action="store_true", help="从 --session-file 恢复会话")
    parser.add_argument("--max-history-pairs", type=int, default=6,
                        help="滑动窗口：保留最近 N 轮对话（默认 6）")
    args = parser.parse_args()

    # 不带 --interactive 且不带 --question 时，默认进入 REPL（更符合"多轮"语义）
    interactive = args.interactive or args.question is None

    # session-file 默认放在项目根
    session_file = args.session_file
    if session_file and not args.resume and not os.path.exists(session_file):
        # 新建会话文件时不强制 resume
        pass

    session = ConversationSession(
        max_history_pairs=args.max_history_pairs,
        session_file=session_file,
    )

    if interactive:
        interactive_repl(session, max_steps=args.max_steps)
    else:
        # 单轮模式（兼容原版 CLI），但仍然会创建 session（无持久化）
        run_and_print(args.question, session, max_steps=args.max_steps)
