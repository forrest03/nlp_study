"""主 Agent + 并行 Subagent 编排（股票看多/看空分析）

架构对应 PPT Part 6.3 的 Orchestrator-Workers 拓扑：
  主 agent 取数据 → 派发 看多/看空 subagent 并行 → 综合双向观点给出最终判定。
"""
import time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from stock_data import get_stock_data, format_stock_brief

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是股票分析主分析师。你有 2 个工具：
- get_stock_data：拉取某公司某日行情（参数=公司|日期，如 "比亚迪|2026-08-04"）
- dispatch_subagents：派发 看多分析师 + 看空分析师 两个子分析师并行分析

【关键决策原则 - 严格按步骤执行，不要重复】
- 第一步（仅一次）：调 get_stock_data 获取公司当日行情。参数格式为 "公司|日期"。
  执行后会得到行情摘要 Observation，其中包含公司名/日期/开收高低/买卖比等信息。
  注意：你只需调用一次 get_stock_data！不要在后续步骤中再次调用。
- 第二步（仅一次）：拿到行情后，调用 dispatch_subagents，参数格式为 "公司|日期"。
  这会派发看多分析师和看空分析师两个子分析师，他们将读取**你刚刚获取的同一份行情数据**，
  分别从正面和负面角度进行分析，确保数据源完全一致。
  注意：你只需派发一次！不要重复 dispatch。
- 第三步：拿到 dispatch_subagents 返回的 Observation（包含两个子分析师的结论）后，
  直接给出 Final Answer 综合判定。不要再次调用任何工具。

【反例 - 禁止这样做】
错误流程：get_stock_data → dispatch_subagents → get_stock_data → dispatch_subagents → Final Answer
这会导致重复调用，浪费时间。正确流程只需：get_stock_data 一次 → dispatch_subagents 一次 → Final Answer

【示例 - 严格按此格式执行】
Question: 查询 比亚迪 在 2026-08-04 的股票，给出多空分析
Thought: 第一步：先获取当日行情
Action: get_stock_data
Action Input: 比亚迪|2026-08-04
Observation: 公司：比亚迪(002594) 日期：2026-08-04 ...（行情摘要）
Thought: 第二步：已获取行情，派发看多分析师和看空分析师并行分析（共享同一份数据）
Action: dispatch_subagents
Action Input: 比亚迪|2026-08-04
Observation: 并行分析完成：2 个子分析师的论据...
Thought: 第三步：已收齐多空双方观点，直接给出综合判定
Final Answer: 综合判定：...（含看多/看空要点 + 最终倾向 + 风险提示）"""


# 看多 / 看空 subagent 的系统提示（引导各自只取正面/负面论据）
BULL_SYSTEM = """你是「看多分析师」（bull analyst），从给定股票行情中提炼**正面看多**论据。

可用工具：
{tools_desc}

规则（严格遵守）：
1. 调用 read_stock_data 读取共享行情（参数任意字符串，如 "read"）
2. 读取到的数据包含【近 5 个交易日行情表格】，你**必须**以表格中的数字作为唯一分析依据
3. 严禁编造数据！所有论据必须引用表格中明确列出的数值（日期、收盘价、涨跌幅等）
4. 从数据中找出支持看多的证据：上涨趋势、放量突破、买盘占优、关键支撑位守住、均线多头排列等
5. 即使整体偏空，也要客观找出可看多的局部信号（不能凭空捏造）
6. 输出 Final Answer：3~5 条看多论据，每条**必须引用表格中具体的日期和数值**，末尾给看多置信度（0-100）

输出格式：
- 每条论据必须包含日期和具体数字，例如："2026-08-07 收盘价 X 元，涨幅 Y%，站上 5 日均线"
- 不允许出现表格中没有的数字
"""

BEAR_SYSTEM = """你是「看空分析师」（bear analyst），从给定股票行情中提炼**负面看空**论据。

可用工具：
{tools_desc}

规则（严格遵守）：
1. 调用 read_stock_data 读取共享行情（参数任意字符串，如 "read"）
2. 读取到的数据包含【近 5 个交易日行情表格】，你**必须**以表格中的数字作为唯一分析依据
3. 严禁编造数据！所有论据必须引用表格中明确列出的数值（日期、收盘价、涨跌幅等）
4. 从数据中找出支持看空的证据：下跌趋势、放量下挫、卖盘占优、破位、高位放量滞涨、均线空头排列等
5. 即使整体偏多，也要客观找出可看空的局部信号（不能凭空捏造）
6. 输出 Final Answer：3~5 条看空论据，每条**必须引用表格中具体的日期和数值**，末尾给看空置信度（0-100）

输出格式：
- 每条论据必须包含日期和具体数字，例如："2026-08-07 收盘价 X 元，跌幅 Y%，跌破 5 日均线"
- 不允许出现表格中没有的数字
"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "{公司}|{日期}"（管道分隔，由主 agent 填）
    派发 看多 + 看空 2 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 解析公司/日期（容错：取主 agent 已拿到的 payload）
    parts = [p.strip() for p in action_input.split("|")]
    company = parts[0] if parts else ""
    date_str = parts[1] if len(parts) > 1 else ""
    # 主 agent 已经把 payload 塞进 shared_state["stock_payload"]
    payload = shared_state.get("stock_payload")
    if payload is None:
        # 兜底：现场再拉一次
        try:
            payload = get_stock_data(company, date_str)
            shared_state["stock_payload"] = payload
        except Exception as e:
            return f"派发失败：获取行情出错 {type(e).__name__}: {str(e)[:120]}"

    brief = format_stock_brief(payload)

    def _read_stock_data(_q, **__):
        # subagent 工具：读取共享行情（不重复联网/拉取）
        return brief

    # 两个固定 subagent：看多 / 看空
    defs = []
    for role, sys_prompt in (("bull", BULL_SYSTEM), ("bear", BEAR_SYSTEM)):
        sid = f"sub_{role}_{uuid.uuid4().hex[:5]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"read_stock_data": (_read_stock_data, "读取共享行情数据，参数任意")},
            max_steps=4, model_tag=f"qwen-plus({role})",
            system_prompt=sys_prompt,
        )
        topic = f"看多分析师({company} {date_str})" if role == "bull" \
            else f"看空分析师({company} {date_str})"
        defs.append((sid, sub, topic, role))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
    dispatch_info = {
        "subtopics": [t for _, _, t, _ in defs],
        "subagent_ids": [sid for sid, _, _, _ in defs],
        "roles": [r for _, _, _, r in defs],
        "company": company, "date": date_str,
    }
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid, sub, topic):
        # subagent 的 question 引导它从行情中提炼对应方向论据
        q = (f"基于以下行情，提炼{('正面看多' if topic.startswith('看多') else '负面看空')}论据。"
             f"公司={company} 日期={date_str}")
        return sid, sub.run(q, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        for sid, sub, topic, _ in defs:
            sid, res = _run_one(sid, sub, topic)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid
                    for sid, sub, topic, _ in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t, _ in defs if s == sid)
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

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【子分析师: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:600]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行分析完成：{len(defs)} 个子分析师，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def _get_stock_data_tool(action_input: str, shared_state: dict = None) -> str:
    """主 agent 工具：拉取股票行情。action_input='公司|日期'。
    把 payload 缓存到 shared_state['stock_payload']，供后续 subagent 共享。"""
    parts = [p.strip() for p in action_input.split("|")]
    if len(parts) < 2:
        return "参数格式错误，应为 '公司|日期'，如 '比亚迪|2026-08-04'"
    company, date_str = parts[0], parts[1]
    try:
        from datetime import datetime
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return f"日期格式错误：{date_str}，应为 YYYY-MM-DD"

    try:
        payload = get_stock_data(company, date_str)
        if shared_state is not None:
            shared_state["stock_payload"] = payload
        return format_stock_brief(payload)
    except Exception as e:
        return f"获取行情失败: {type(e).__name__}: {str(e)[:160]}"


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次股票多空分析。返回 {final_answer, main_trace, subagents, parallel_stats, stock_payload}。
    serial=True 时 subagent 串行执行（eval A/B 对比基线）。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    main = ReActLoop(
        agent_name="main",
        tools={
            "get_stock_data": (_get_stock_data_tool,
                               "获取股票当日行情，参数=公司|日期（如 比亚迪|2026-08-04）"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发 看多 + 看空 两个子分析师并行分析，参数=公司|日期"),
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
        "stock_payload": shared_state.get("stock_payload"),
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "查询 比亚迪 在 2026-08-04 的股票，给出多空分析"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n最终判定头:\n{r['final_answer'][:300]}")
