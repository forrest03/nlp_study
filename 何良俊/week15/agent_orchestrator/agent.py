"""
agent — 通用 Agent 决策循环核心
=================================

编排 Agent 与 subagent 共用同一个 Agent 类，只是「角色提示词 + 工具集 + 步数上限」
不同。这就是本项目最核心的抽象，一个 agent 的完整组件流程：

    系统提示词 → LLM 推理 → 决策(是否调用工具?) → 执行工具 / 回传观察 → 最终答案

决策循环（函数调用版 ReAct，对应真实世界 Agent 的 run loop）：
    1. 调用 LLM（附工具 schema）
    2. 若返回 tool_calls → 逐个执行工具，把结果作为 tool 消息回传，回到步骤 1
    3. 若无 tool_calls → 本次回复即最终答案，循环结束
    4. 超过 max_steps 仍未结束 → 强制结束并标记失败

循环中每一步通过 on_event 回调实时上报（交给 display 展示），自身不打印。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from llm import DeepSeekClient


@dataclass
class Tool:
    """一个可注册给 agent 的工具：schema（给 LLM 看）+ fn（真正执行）。"""

    name: str
    description: str
    parameters: dict          # JSON Schema（OpenAI tools 格式）
    fn: Callable

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class StepRecord:
    """循环中的一步记录（供展示与落盘）。"""

    step: int
    kind: str                 # "action" | "final" | "error"
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    answer: str = ""


class Agent:
    """通用决策循环 agent。"""

    def __init__(
        self,
        name: str,
        role: str,                       # 系统提示词（角色设定）
        client: DeepSeekClient,
        tools: list[Tool],
        max_steps: int = 8,
        observation_limit: int = 2000,   # 回传 LLM 的观察截断长度（保护上下文）
        on_event: Optional[Callable[[dict], None]] = None,
        stream_final: bool = True,       # 最终答案是否流式输出（subagent 关掉，便于整块查看）
    ):
        self.name = name
        self.role = role
        self.client = client
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.observation_limit = observation_limit
        self.stream_final = stream_final
        self._emit = on_event or (lambda _ev: None)

    def run(self, question: str) -> dict:
        """执行完整决策循环，返回结果 dict（最终答案 + 步骤轨迹 + 统计）。"""
        messages: list[dict] = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": question},
        ]
        trace: list[StepRecord] = []
        final = f"（未能在 {self.max_steps} 步内得出最终答案）"
        failed = False
        t0 = time.time()

        for step in range(1, self.max_steps + 1):
            # 流式开启时：思考（reasoning_content）与文本 token 实时上报
            # （thought_delta / final_delta 事件），tool_calls 增量自动合并。
            resp = self.client.chat_with_tools(
                messages, [t.schema() for t in self.tools.values()],
                stream=self.stream_final,
                on_text_delta=lambda d: self._emit(
                    {"type": "final_delta", "agent": self.name, "delta": d}
                ) if self.stream_final else None,
                on_reasoning_delta=lambda d: self._emit(
                    {"type": "thought_delta", "agent": self.name, "step": step, "delta": d}
                ) if self.stream_final else None,
                # 流中探测到 tool_call：展示层据此区分"决策前言"与"最终答案"
                on_tool_call_seen=lambda: self._emit(
                    {"type": "tool_call_seen", "agent": self.name}
                ) if self.stream_final else None,
            )

            # 思考过程 = ReAct 的 Thought（函数调用模式下模型思维链在
            # reasoning_content 字段）。记录进 trace；非流式时整条上报。
            if resp.reasoning:
                trace.append(StepRecord(step=step, kind="thought", answer=resp.reasoning))
                if not self.stream_final:
                    self._emit({
                        "type": "thought", "agent": self.name,
                        "step": step, "thought": resp.reasoning,
                    })

            # 模型决定直接回答 → 循环结束
            if not resp.tool_calls:
                final = resp.text or "（模型返回空内容）"
                trace.append(StepRecord(step=step, kind="final", answer=final))
                self._emit({
                    "type": "final", "agent": self.name,
                    "step": step, "answer": final, "streamed": self.stream_final,
                })
                break

            # 模型请求调用工具：把这条 assistant 消息并入对话（API 要求）
            messages.append(resp.message)

            for tc in resp.tool_calls:
                fn = tc.get("function") or {}
                fn_name = fn.get("name", "")
                try:
                    fn_args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(fn_args, dict):
                        fn_args = {}
                except json.JSONDecodeError:
                    fn_args = {}          # 参数解析失败降级为空 dict

                tool = self.tools.get(fn_name)
                # 先上报"决策 → 行动"再执行工具：长耗时工具（如并行下发
                # subagent）执行期间，用户能看到正在做什么（ReAct 的 Action）
                trace.append(StepRecord(
                    step=step, kind="action", action=fn_name, action_input=fn_args,
                ))
                self._emit({
                    "type": "action",
                    "agent": self.name,
                    "step": step,
                    "action": fn_name,
                    "action_input": fn_args,
                })

                if tool is None:
                    observation = f"未知工具: {fn_name}"
                else:
                    try:
                        observation = tool.fn(**fn_args)
                    except TypeError as e:
                        observation = f"工具参数错误: {e}"
                    except Exception as e:
                        observation = f"工具执行失败: {e}"
                observation = str(observation)
                trace[-1].observation = observation
                self._emit({
                    "type": "observation",
                    "agent": self.name,
                    "step": step,
                    "action": fn_name,
                    "observation": observation,
                })

                # 观察截断后回传 LLM，避免长结果撑爆上下文（完整内容在 trace 里）
                limit = self.observation_limit
                if len(observation) > limit:
                    observation = (
                        observation[:limit]
                        + f"\n…（观察过长，已截断，原长 {len(observation)} 字符）"
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": observation,
                })
        else:
            # 达到 max_steps 仍未得到最终答案
            failed = True
            trace.append(StepRecord(step=self.max_steps, kind="error", answer=final))
            self._emit({"type": "error", "agent": self.name, "answer": final})

        return {
            "name": self.name,
            "question": question,
            "final": final,
            "trace": [vars(s) for s in trace],
            "steps": len(trace),
            "duration_s": round(time.time() - t0, 2),
            "failed": failed,
        }
