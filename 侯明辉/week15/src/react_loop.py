"""
通用 ReAct 循环引擎。

ReAct 格式:
  Thought: <推理>
  Action: <工具名>
  Action Input: <参数字符串>
  ...（runner 执行工具后）Observation: <结果>
  ... 直到
  Thought: 信息充分
  Final Answer: <答案>

stop=["Observation:"] 让 LLM 在 Action Input 后停下。

依赖：re, time, llm_client.llm_chat
"""
import re
import time
import logging
from typing import Callable

from llm_client import llm_chat

logger = logging.getLogger(__name__)


def _build_tools_desc(tools: dict) -> str:
    """tools: {name: (fn, description_str)} → 工具说明文本"""
    lines = [f"- {name}: {desc}" for name, (_, desc) in tools.items()]
    return "\n".join(lines)


class ReActLoop:
    """通用 ReAct 循环。主 agent / subagent 各自实例化一个。"""

    DEFAULT_SYSTEM = """你是 ReAct agent。按如下格式严格输出（每轮一次）：

Thought: 你的推理
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步"""

    def __init__(self, agent_name: str, tools: dict,
                 system_prompt: str | None = None,
                 max_steps: int = 6):
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.system = system_prompt or self.DEFAULT_SYSTEM
        self.trace: list[dict] = []

    def run(self, question: str, on_step: Callable | None = None,
            extra_obs: str | None = None) -> dict:
        """
        执行 ReAct 循环。返回 {final_answer, trace, duration}。

        on_step(step_dict): 每步回调（调试/可视化用）
        extra_obs: 在第一轮 Thought 前附加额外上下文（dispatch 用）
        """
        self.trace = []
        t0 = time.time()

        # 注入工具描述到 system prompt
        tools_desc = _build_tools_desc(self.tools)
        system = self.system + "\n\n可用工具：\n" + tools_desc

        history = f"Question: {question}\n"
        if extra_obs:
            history += f"\n{extra_obs}\n"
        history += "\n"

        final_answer = ""
        for step_idx in range(self.max_steps):
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=768, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": None}

            if action == "Final Answer":
                if not action_input.strip():
                    # 空 Final Answer 视为格式错误，附加 observation 让 LLM 重试
                    observation = "你的 Final Answer 是空的，请重新调用工具或写出完整答案"
                    step["observation"] = observation
                    step["done"] = True
                    self.trace.append(step)
                    if on_step: on_step(step)
                    history += llm_out + f"Observation: {observation}\n"
                    continue  # 不 break，继续下一轮
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step: on_step(step)
                break

            # 执行工具
            observation = self._exec_tool(action, action_input)
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step: on_step(step)

            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            # 超 max_steps 强制收尾
            last_obs = self.trace[-1].get("observation", "") if self.trace else ""
            final_answer = f"（已达最大步数）{last_obs[:300]}"
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": None, "final": True}
            self.trace.append(step)
            if on_step: on_step(step)

        return {"final_answer": final_answer, "trace": self.trace,
                "duration": round(time.time() - t0, 2)}

    def _parse(self, text: str) -> tuple[str, str, str]:
        """从 LLM 输出解析 Thought/Action/Action Input。"""
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        # Final Answer 优先
        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = mi.group(1).strip() if mi else ""
            return thought, action, action_input

        # 兜底：有文本无格式 → 当 Final Answer
        if text.strip():
            return thought or "综合结果给出报告", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str) -> str:
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            return str(fn(action_input))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    # 自测：单工具 ReAct 跑通
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent))
    from tools import read_file

    loop = ReActLoop(
        agent_name="test",
        tools={"read_file": (read_file, "读取文件片段，参数 path:start:end")},
        max_steps=4,
    )
    q = "week15homework_agent 的 requirements.txt 第一行是什么"
    r = loop.run(q)
    print(f"\n答案: {r['final_answer'][:200]}")
    print(f"trace {len(r['trace'])} 步:")
    for s in r["trace"]:
        print(f"  [{s['idx']}] {s['action']}({(s.get('action_input') or '')[:50]!r})")
    assert r["final_answer"], "final_answer 为空"
    assert any("openai" in (s.get("observation") or "") for s in r["trace"]), \
        "未读到 requirements.txt 内容"
    print("\n✓ react_loop 自测通过")
