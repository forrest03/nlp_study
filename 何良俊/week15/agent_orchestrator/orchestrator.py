"""
orchestrator — 编排 Agent：拆解任务 → 并行下发 subagent → 收集 → 汇总
======================================================================

编排者是一个特殊 Agent：
- 角色提示词要求它把用户任务拆解成相互独立的子任务
- 它唯一的工具是 dispatch_subagents：一次下发多个 subagent 并行执行
- 工具执行期间用 ThreadPoolExecutor 并发运行每个 subagent
  （每个 subagent 都是独立的 Agent：自己的角色提示词 + 小工具集 + 决策循环）
- 全部完成后：
    * 完整结果经事件回调交给 display 展示给用户
    * 压缩版摘要作为 Observation 回传给编排 LLM，由它汇总最终答案
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from agent import Agent, Tool
from llm import DeepSeekClient
from tools import calculator, current_date

# ── 角色提示词 ────────────────────────────────────────────────────────────────

ORCHESTRATOR_ROLE = """你是任务编排 Agent（orchestrator），负责把用户的复杂任务拆解为多个相互独立的子任务，
并行下发给 subagent 执行，再汇总成最终答案。

工作流程：
1. 分析用户任务，判断是否可以拆分为多个相互独立、可并行的子任务
2. 如果可以拆分，必须调用 dispatch_subagents 一次性下发全部子任务
   （2~5 个为宜，全部放进同一次调用的 tasks 数组）
3. 收到全部子任务结果后，汇总并输出结构化的最终答案

规则：
- 工具纪律：calculator / current_date 只在确有需要时使用
  （真的要做数值计算、真的依赖今天日期），否则不要调用
- 简单任务直接回答；能自己用工具完成的直接完成，不要为了用 subagent 而用
- 只有任务可以拆分为多个相互独立、可并行的子任务，且拆分并行能明显提升
  质量或效率时，才调用 dispatch_subagents
- 重要：所有子任务必须在同一次 dispatch_subagents 调用中下发，
  禁止拆成多次调用（多次调用会把并行任务串行化，且浪费决策轮次）
- 子任务必须相互独立：每个 subagent 只能看到自己的 goal 与 context，
  不能依赖其他子任务的结果（它们之间没有通信机制）
- 每个子任务的 goal 必须自包含、具体、可执行，并明确要求输出格式；
  除非调研主题本身涉及量化指标，否则不要在 goal 里要求数值计算
- 最终答案必须基于子任务结果，可以整合、对比、去重、补充结论
"""

SUBAGENT_ROLE_TEMPLATE = """你是「{name}」子任务 Agent，隶属于一个多 Agent 协作系统。

你只负责下面这一项任务，不要越界：
{goal}

规则（工具纪律，务必遵守）：
- 这是一个研究/分析类任务：**默认直接基于你的知识作答，不调用任何工具**
- calculator 只在任务确实需要具体数值计算（如百分比、倍率、成本测算）时才调用，
  算过一次、结果够用就直接用它作答，不要重复或连环调用
- current_date 只在回答确实依赖"今天日期"时才调用
- 工具调用是手段不是目的：没有数值计算需求时，调用工具就是错误的
- 输出必须是完整、自包含的最终答案（可用 Markdown 结构化），
  不要提及"其他 agent"等协作细节
- 不要尝试与其他 agent 通信
{context_block}"""

# ── 通用工具集：主 agent 与 subagent 共同具备的能力 ─────────────────────────
# subagent = 主 agent 的能力全集 - dispatch_subagents（不能再下发）

COMMON_TOOLS = [
    Tool(
        name="calculator",
        description=(
            "安全计算数学表达式，返回计算结果字符串。"
            "支持 + - * / // % ** 括号、常量 pi/e、"
            "函数 abs/round/min/max/sqrt/pow/floor/ceil。"
            "注意：仅在任务确实需要数值计算时才调用，不要为了调用而调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，如 (2*30.5+100)/4",
                },
            },
            "required": ["expression"],
        },
        fn=calculator,
    ),
    Tool(
        name="current_date",
        description="返回今天的日期（YYYY-MM-DD 格式）。",
        parameters={"type": "object", "properties": {}},
        fn=current_date,
    ),
]

# ── 编排器 ───────────────────────────────────────────────────────────────────

MAX_TASKS_PER_DISPATCH = 8
SUMMARY_FINAL_LIMIT = 1500   # 摘要回传 LLM 时每个结果 final 的截断长度


class OrchestratorAgent:
    """编排 Agent：拆解 → 并行下发 → 收集 → 汇总。"""

    def __init__(
        self,
        client: DeepSeekClient,
        max_workers: int = 3,
        subagent_max_steps: int = 6,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        self.client = client
        self.max_workers = max_workers
        self.subagent_max_steps = subagent_max_steps
        self._emit = on_event or (lambda _ev: None)
        self.last_dispatch: list[dict] = []   # 最近一次下发的完整结果（供展示/落盘）
        self._task_seq = 0                    # 全局任务计数器：跨多次下发 id 不重名
        self._core = Agent(
            name="orchestrator",
            role=ORCHESTRATOR_ROLE,
            client=client,
            # 主 agent = 通用工具集 + dispatch_subagents（唯一超出 subagent 的能力）
            tools=[*COMMON_TOOLS, self._dispatch_tool()],
            max_steps=12,
            observation_limit=4000,
            on_event=on_event,
        )

    def run(self, question: str) -> dict:
        """执行编排者的完整决策循环，返回最终结果 dict。"""
        return self._core.run(question)

    # ---- dispatch_subagents 工具 ----------------------------------------

    def _dispatch_tool(self) -> Tool:
        return Tool(
            name="dispatch_subagents",
            description=(
                "把当前任务拆分为多个相互独立、可并行的子任务，"
                "并行下发给 subagent 执行，全部完成后返回各子任务的结果摘要。"
                "必须一次调用下发全部子任务（全部放进 tasks 数组），"
                "不要分多次调用本工具。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": (
                            "子任务列表，2~5 个为宜；每个子任务的 goal 必须自包含，"
                            "subagent 只能看到自己的 goal 与 context"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "子任务编号，如 task-1、task-2"},
                                "name": {"type": "string", "description": "子任务名称，如「推理性能优化调研」"},
                                "goal": {"type": "string", "description": "子任务目标：要求子任务完成什么、输出什么格式"},
                                "context": {"type": "string", "description": "可选的背景材料/上下文"},
                            },
                            "required": ["id", "name", "goal"],
                        },
                    },
                },
                "required": ["tasks"],
            },
            fn=self._dispatch_subagents,
        )

    # ---- 工具实现 ---------------------------------------------------------

    def _dispatch_subagents(self, tasks) -> str:
        """并行执行所有子任务。返回给 LLM 的观察为压缩摘要，完整结果走事件。"""
        tasks = self._normalize_tasks(tasks or [])
        if not tasks:
            return "错误: 未收到任何子任务，请检查 tasks 参数"

        self._emit({"type": "dispatch_start", "tasks": tasks})

        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {t["id"]: pool.submit(self._run_one_subagent, t) for t in tasks}
            for t in tasks:                       # 按原顺序取回，展示更稳定
                fut = futures[t["id"]]
                try:
                    results[t["id"]] = fut.result()
                except Exception as e:            # 子线程内部异常也要兜底展示
                    results[t["id"]] = {
                        "id": t["id"], "name": t["name"], "goal": t["goal"],
                        "status": "failed", "error": str(e), "final": "",
                        "trace": [], "steps": 0, "duration_s": 0, "failed": True,
                    }

        ordered = [results[t["id"]] for t in tasks]
        self.last_dispatch = ordered
        self._emit({"type": "dispatch_report", "results": ordered})

        # 压缩摘要回传 LLM（每个结果只保留状态 + 截断后的最终答案）
        summary = []
        for r in ordered:
            final = (r.get("final") or r.get("error") or "").strip()
            summary.append({
                "id": r["id"],
                "name": r["name"],
                "status": "ok" if not r.get("failed") else "failed",
                "duration_s": r.get("duration_s"),
                "final": final[:SUMMARY_FINAL_LIMIT],
            })
        return json.dumps(summary, ensure_ascii=False, indent=1)

    def _normalize_tasks(self, tasks: list) -> list[dict]:
        """容错：id 用全局计数器生成（跨多次下发不重名），过滤无效项、上限 MAX_TASKS。"""
        out: list[dict] = []
        for t in tasks:
            if len(out) >= MAX_TASKS_PER_DISPATCH:
                break
            if not isinstance(t, dict) or not str(t.get("goal") or "").strip():
                continue
            self._task_seq += 1
            out.append({
                "id": f"task-{self._task_seq}",
                "name": str(t.get("name") or f"子任务{self._task_seq}"),
                "goal": str(t["goal"]).strip(),
                "context": str(t.get("context") or "").strip(),
            })
        return out

    # ---- subagent 执行 -----------------------------------------------------

    def _run_one_subagent(self, task: dict) -> dict:
        """运行一个独立 subagent（自带角色提示词 + 小工具集 + 决策循环）。"""
        self._emit({"type": "subagent_start", "task": task})

        context_block = (
            f"\n背景材料:\n{task['context']}" if task["context"] else ""
        )
        role = SUBAGENT_ROLE_TEMPLATE.format(
            name=task["name"], goal=task["goal"], context_block=context_block,
        )

        subagent = Agent(
            name=task["name"],
            role=role,
            client=self.client,
            tools=COMMON_TOOLS,   # 通用工具集，无 dispatch_subagents → 不能再下发
            max_steps=self.subagent_max_steps,
            observation_limit=1500,
            stream_final=False,   # subagent 一次性输出完整结果，便于整块查看
            # 子线程事件统一转发，并带上任务归属信息供展示
            on_event=lambda ev: self._emit({
                **ev,
                "subagent_id": task["id"],
                "subagent_name": task["name"],
            }),
        )

        result = subagent.run(task["goal"])
        result["id"] = task["id"]
        result["goal"] = task["goal"]
        result["status"] = "failed" if result["failed"] else "ok"
        self._emit({"type": "subagent_done", "result": result})
        return result
