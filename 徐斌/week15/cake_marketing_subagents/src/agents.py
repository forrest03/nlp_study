"""
蛋糕商品采集 + 营销设计：主 Agent + 并行 Subagent

范式：动态 Orchestrator-Workers
  - 主 agent：web_search + dispatch_subagents，自主拆分课题
  - subagent：仅 web_search（含图片），并行采集 / 分析 / 营销设计素材

典型派发侧面：
  1) 蛋糕商品详情（名称/价格/规格/文字介绍/图片 URL）
  2) 竞品卖点与用户场景
  3) 营销设计（定位、文案、活动、视觉建议）
"""
from __future__ import annotations

import time
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from browser_search import browser_web_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是「蛋糕商品采集与营销设计」主策划。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子 agent 并行工作（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要任务涉及「采集商品详情（图片/文字）」和/或「营销设计/文案/活动」等 2 个及以上侧面，
  必须用 dispatch_subagents 拆给子 agent 并行处理，不要自己串行 web_search 多次。
- 只有单一事实（如「某品牌某款蛋糕现价」）才直接 web_search
- 拿到子结果后，综合成结构化交付物

【建议拆分（可按用户问题微调，通常 3 个）】
1. 蛋糕商品详情采集：商品名、价格区间、规格口味、文字介绍、图片 URL、来源链接
2. 竞品与用户场景：主流品牌/SKU、卖点、适用场景（生日/下午茶/节日）、促销手法
3. 营销设计方案：定位一句话、目标人群、主视觉建议、卖点文案、活动玩法、短视频/海报提纲

【最终报告结构（必须遵守）】
## 一、商品详情清单
（每款：名称 | 价格 | 规格/口味 | 文字介绍摘要 | 图片URL | 来源）
## 二、竞品与场景洞察
## 三、营销设计方案
（定位 / 人群 / 视觉方向 / 文案 / 活动 / 渠道建议）
## 四、结论与不确定性

【示例】
Question: 采集生日蛋糕类商品详情（图片+文字介绍），并给出营销设计方案
Thought: 这是多侧面任务（商品采集 + 竞品洞察 + 营销设计），必须派发子 agent 并行
Action: dispatch_subagents
Action Input: 生日蛋糕热销商品详情：名称价格规格文字介绍与图片URL | 生日蛋糕竞品卖点与用户购买场景 | 生日蛋糕营销设计：定位文案活动与视觉建议
Observation: 并行完成...
Thought: 已收齐三路结果，综合成报告
Final Answer: （按上述四段结构输出）"""

SUB_SYSTEM = """你是蛋糕调研子 agent，专注完成主策划分配的单一子课题。

可用工具：
{tools_desc}

按如下格式严格输出：
Thought: ...
Action: 工具名
Action Input: 查询词

最后：
Thought: 信息足够
Final Answer: 针对本子课题的结构化答案

专项要求：
- 若课题要求商品详情：尽量列出 3~6 款；每款包含名称、价格(若有)、规格/口味、文字介绍要点、图片 URL、来源 URL
- 若课题是营销设计：输出可落地的定位句、3 条卖点文案、1 套活动玩法、视觉关键词（色调/材质/构图）
- 图片 URL、商品名、价格必须以 Observation 原文为准，禁止编造不存在的链接或虚构品牌 SKU
- 每轮只调一次工具"""


def _dispatch_subagents(
    action_input: str,
    shared_state: dict = None,
    on_subagent_step: Callable = None,
    on_subagent_done: Callable = None,
    on_dispatch: Callable = None,
    serial: bool = False,
) -> str:
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={
                "web_search": (
                    lambda q, **_: format_search_result(
                        browser_web_search(q, include_images=True)
                    ),
                    "模拟浏览器联网搜索（含图片URL），参数是查询词",
                )
            },
            max_steps=5,
            model_tag="qwen-plus(子)",
            system_prompt=SUB_SYSTEM,
        )
        defs.append((sid, sub, topic))

    dispatch_info = {
        "subtopics": subtopics,
        "subagent_ids": [sid for sid, _, _ in defs],
    }
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid=None, sub=None, topic=None):
        return sid, sub.run(
            topic,
            on_step=(
                lambda step, sid=sid: on_subagent_step(sid, step)
                if on_subagent_step
                else None
            ),
        )

    if serial:
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic,
                "trace": res["trace"],
                "duration": res["duration"],
                "final_answer": res["final_answer"],
            }
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {
                pool.submit(_run_one, sid, sub, topic): sid
                for sid, sub, topic in defs
            }
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic,
                    "trace": res["trace"],
                    "duration": res["duration"],
                    "final_answer": res["final_answer"],
                }
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append(
        {
            "n_subagents": len(defs),
            "wall_clock": wall,
            "serial_sum": serial_sum,
            "speedup": round(serial_sum / wall, 2) if wall else 0,
        }
    )

    # 营销/商品详情内容较长，单子结果截到 800 字喂回主 agent
    parts = [
        f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:800]}"
        for sid, (topic, r) in results.items()
    ]
    stats = shared_state["parallel_stats"][-1]
    return (
        f"并行完成：{len(defs)} 个子 agent，wall-clock {wall}s "
        f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n"
        + "\n\n".join(parts)
    )


def run_cake_research(
    question: str,
    on_main_step: Callable = None,
    on_subagent_step: Callable = None,
    on_subagent_done: Callable = None,
    on_dispatch: Callable = None,
    serial: bool = False,
) -> dict:
    """执行一次蛋糕采集+营销任务。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return _dispatch_subagents(
            action_input,
            shared_state=info,
            on_subagent_step=on_subagent_step,
            on_subagent_done=on_subagent_done,
            on_dispatch=on_dispatch,
            serial=serial,
        )

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (
                lambda q, **_: format_search_result(
                    browser_web_search(q, include_images=True)
                ),
                "模拟浏览器联网搜索一次（含图片），参数=查询词",
            ),
            "dispatch_subagents": (
                dispatch_tool,
                "派发多个子 agent 并行，参数=用 | 分隔的多个子课题",
            ),
        },
        max_steps=8,
        model_tag="qwen-plus(主)",
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


# 兼容旧入口名
run_research = run_cake_research


if __name__ == "__main__":
    import logging as _l

    _l.basicConfig(level=_l.WARNING)
    q = (
        "采集生日蛋糕类商品详情（图片URL + 文字介绍 + 价格规格），"
        "并基于竞品给出营销设计方案（定位/文案/活动/视觉）"
    )
    r = run_cake_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:400]}")
