"""
股票分析 Agent — 并行 Subagent 编排

功能：输入股票代码或名称，并行完成 3 项调研：
  1. 公司相关信息（公司概况、主营业务、市值等）
  2. 股票最近走势（股价变动、成交量、技术指标等）
  3. 公司最近新闻（重大公告、行业政策、事件动态等）

架构：与 agents.py 相同的 Orchestrator-Workers 拓扑
  - 主 agent 是 ReAct 循环，有 web_search + dispatch_subagents 工具
  - 派发 3 个 subagent 并行调研，ThreadPoolExecutor 加速
  - 每个 subagent 也是 ReAct 循环（只有 web_search 工具）
  - wall-clock ≈ max(单 agent 时长)，而非 sum

使用方式：
  from stock_agent import analyze_stock
  r = analyze_stock("贵州茅台")
  print(r["final_answer"])
"""

import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

STOCK_SYSTEM = """你是股票分析主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如股票分析的公司信息、走势、新闻等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
- 只有单一事实问题才直接 web_search
- 拿到子调研结果后，综合成结构化报告

【股票分析标准子课题拆分】
当用户输入股票代码或名称时，应派发以下 3 个子课题（用 | 分隔）：
  1. "{{股票名}} 公司概况：主营业务、成立时间、总部、上市情况、市值"
  2. "{{股票名}} 股票近期走势：近3个月股价变动、成交量趋势、技术指标"
  3. "{{股票名}} 公司近期新闻：最近1个月重大公告、行业政策、事件动态"

报告要求：分维度组织（公司概况 / 走势分析 / 新闻与事件），每个要点带来源，末尾给投资建议与风险提示。

【示例】
Question: 贵州茅台 股票分析
Thought: 这是股票分析，涉及公司信息、走势、新闻三个侧面，必须并行派发
Action: dispatch_subagents
Action Input: 贵州茅台 公司概况：主营业务、成立时间、总部、上市情况、市值 | 贵州茅台 股票近期走势：近3个月股价变动、成交量趋势 | 贵州茅台 公司近期新闻：最近1个月重大公告、行业政策
Observation: 并行调研完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三个维度的并行调研结果，综合成报告
Final Answer: （分维度报告，含投资建议与风险提示）"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。"""
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
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索，参数是查询词")},
            max_steps=4, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, topic))

    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid=sid, sub=sub, topic=topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            topic = next(t for s, _, t in defs if s == sid)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def analyze_stock(stock_input: str, on_main_step: Callable = None,
                  on_subagent_step: Callable = None,
                  on_subagent_done: Callable = None,
                  on_dispatch: Callable = None,
                  serial: bool = False) -> dict:
    """执行股票分析。输入可以是股票代码（如 600519）或股票名称（如 贵州茅台）。
    返回 {final_answer, main_trace, subagents, parallel_stats, dispatches}。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    question = f"请对【{stock_input}】进行全面的股票分析，包括公司信息、近期走势和最近新闻。"

    main = ReActLoop(
        agent_name="stock_main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子调研员并行调研，参数=用 | 分隔的多个子课题"),
        },
        max_steps=8,
        model_tag="deepseek-chat(股票主)",
        system_prompt=STOCK_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    stock = input("请输入股票代码或名称（如 600519 或 贵州茅台）：").strip() or "贵州茅台"

    def on_main_step(step):
        tag = "★" if step.get("final") else "→"
        obs = (step.get("observation") or "")[:60]
        extra = f" | {obs}" if obs else ""
        print(f"  [主Agent] {tag} {step['action']}: {step['action_input'][:80]}{extra}")

    def on_dispatch(info):
        topics_preview = [t[:30] for t in info['subtopics']]
        print(f"\n{'─'*50}")
        print(f"  [派发] {len(info['subagent_ids'])} 个子agent: {topics_preview}")
        print(f"  [并行] 子agent正在并行执行，请稍候...")

    def on_subagent_step(sid, step):
        if not step.get("observation") and not step.get("final"):
            return
        tag = "★" if step.get("final") else "→"
        obs = (step.get("observation") or "")[:50]
        extra = f" | {obs}" if obs else ""
        print(f"    [{sid}] {tag} {step['action']}: {step['action_input'][:60]}{extra}")

    def on_subagent_done(sid, duration, topic):
        print(f"    [{sid}] ✓ 完成 ({duration}s) - {topic[:40]}")

    print(f"\n{'='*60}")
    print(f"  股票分析: {stock}")
    print(f"  主 Agent 正在规划任务...")
    t0 = time.time()
    r = analyze_stock(stock,
                      on_main_step=on_main_step,
                      on_dispatch=on_dispatch,
                      on_subagent_step=on_subagent_step,
                      on_subagent_done=on_subagent_done)
    total = round(time.time() - t0, 2)

    print(f"\n{'─'*50}")
    print(f"  总耗时: {total}s")
    print(f"  主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"  派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    if r["parallel_stats"]:
        print(f"  并行统计: {r['parallel_stats']}")
    print(f"\n{'='*60}")
    print(f"最终报告:\n")
    print(r["final_answer"])