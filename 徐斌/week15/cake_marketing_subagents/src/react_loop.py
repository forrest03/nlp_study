"""
通用 ReAct 循环引擎

主 agent 与 subagent 共用；区别只在 tools 字典与 system_prompt。
"""
from __future__ import annotations

import time
import re
import logging
from typing import Callable, Optional

from llm_client import llm_chat

logger = logging.getLogger(__name__)

REACT_SYSTEM = """你是蛋糕商品调研助手，能用以下工具联网搜索。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（带来源要点；涉及商品时尽量保留图片 URL 与文字介绍）

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步"""


def build_tools_desc(tools: dict) -> str:
    lines = []
    for name, (fn, desc) in tools.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


class ReActLoop:
    """通用 ReAct 循环。主 agent / subagent 各自实例化一个。"""

    def __init__(
        self,
        agent_name: str,
        tools: dict,
        max_steps: int = 6,
        model_tag: str = "llm",
        system_prompt: Optional[str] = None,
    ):
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM
        self.trace: list[dict] = []

    def run(
        self,
        question: str,
        on_step: Callable = None,
        shared_state: dict = None,
    ) -> dict:
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"
        final_answer = ""

        for step_idx in range(self.max_steps):
            llm_out = llm_chat(
                system,
                history,
                temperature=0.0,
                max_tokens=1024,
                stop=["Observation:"],
            )
            thought, action, action_input = self._parse(llm_out)

            step = {
                "idx": step_idx,
                "agent": self.agent_name,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": None,
            }

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            step["final"] = False
            if on_step:
                on_step(step)

            # 空 Action：不调工具，提示按格式重试，避免无效步浪费
            if not (action or "").strip():
                observation = (
                    "未检测到有效 Action。请严格输出 Thought/Action/Action Input，"
                    "或在信息足够时输出 Final Answer。"
                )
            else:
                observation = self._exec_tool(action, action_input, shared_state)

            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step:
                on_step(step)

            history += llm_out + f"Observation: {observation[:1400]}\n"

        else:
            final_answer = "（已达最大步数）" + (
                self.trace[-1].get("observation", "") or ""
            )
            step = {
                "idx": self.max_steps,
                "agent": self.agent_name,
                "thought": "达到步数上限",
                "action": "Final Answer",
                "action_input": final_answer,
                "observation": None,
                "final": True,
            }
            self.trace.append(step)
            if on_step:
                on_step(step)

        duration = round(time.time() - t0, 2)
        return {
            "final_answer": final_answer,
            "trace": self.trace,
            "duration": duration,
        }

    def _parse(self, text: str) -> tuple[str, str, str]:
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = mi.group(1).strip() if mi else ""
            return thought, action, action_input

        if text.strip():
            return thought or "综合结果给出报告", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            return str(
                fn(action_input, shared_state=shared_state)
                if shared_state is not None
                else fn(action_input)
            )
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    import logging as _l

    _l.basicConfig(level=_l.WARNING)
    from browser_search import browser_web_search, format_search_result

    def web_search(q, **_):
        return format_search_result(browser_web_search(q))

    loop = ReActLoop(
        "test",
        tools={"web_search": (web_search, "联网搜索，参数是查询词")},
        max_steps=4,
    )
    r = loop.run("芝士蛋糕 经典口味 商品介绍")
    print(f"\n答案: {r['final_answer'][:200]}")
