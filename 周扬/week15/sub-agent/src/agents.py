"""通用主 Agent + 并行子 Agent 调度层。"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.react_loop import ReActLoop
from src.tavily_search import format_search_result, tavily_search


MAIN_SYSTEM = """你是通用任务协调 Agent。你的可用工具如下：
{tools}

决策规则：
1. 对能一次检索回答的简单事实或单一问题，调用 web_search。
2. 对包含两个及以上可独立处理的部分的问题，调用 dispatch_subagents。
3. dispatch_subagents 的参数必须是 2 到 4 个完整、可独立执行的子任务，用 | 分隔。
4. 收到子 Agent 的 Observation 后，整合为清晰、可执行的最终答案。
5. 子 Agent 只能检索资料，不能继续派发任务。

示例：
Question: 帮我制定去杭州三天的旅行计划，需要交通、住宿区域和景点安排
Thought: 这个任务可以拆成交通、住宿区域和景点安排三个独立部分，应并行派发子 Agent。
Action: dispatch_subagents
Action Input: 杭州三天旅行的交通建议 | 杭州住宿区域选择建议 | 杭州三天景点安排建议

工具返回 Observation 后，必须输出：
Thought: 已收齐子任务结果
Final Answer: 结合各部分结果给出完整方案
"""


def search_tool(query, shared_state=None):
    return format_search_result(tavily_search(query))


def dispatch_subagents(action_input, shared_state=None):
    """主 Agent 的第二个工具：创建只会搜索的执行 Agent 并行处理子任务。"""
    topics = [item.strip() for item in action_input.split("|") if item.strip()][:4]
    if len(topics) < 2:
        return "至少需要两个用 | 分隔的子任务。"

    state = shared_state if shared_state is not None else {}
    state.setdefault("subagents", {})
    callback = state.get("callbacks", {})
    definitions = []
    for topic in topics:
        agent_id = "sub_" + uuid.uuid4().hex[:6]
        subagent = ReActLoop(
            agent_id,
            {"web_search": (search_tool, "联网搜索一次，参数是查询词")},
            max_steps=3,
        )
        definitions.append((agent_id, topic, subagent))

    if callback.get("on_dispatch"):
        callback["on_dispatch"]({"subtopics": topics, "subagent_ids": [item[0] for item in definitions]})

    start_time = time.time()
    results = {}

    def run_one(agent_id, topic, subagent):
        def on_step(step):
            if callback.get("on_subagent_step"):
                callback["on_subagent_step"](agent_id, step)
        return agent_id, topic, subagent.run(topic, on_step=on_step)

    if state.get("serial"):
        # 对比基线：子 Agent 一个一个执行。
        for definition in definitions:
            agent_id, topic, result = run_one(*definition)
            results[agent_id] = {"subtopic": topic, **result}
            if callback.get("on_subagent_done"):
                callback["on_subagent_done"](agent_id, result["duration"], topic)
    else:
        # 正常模式：多个互不依赖的子任务并行搜索。
        with ThreadPoolExecutor(max_workers=len(definitions)) as executor:
            futures = [executor.submit(run_one, *item) for item in definitions]
            for future in as_completed(futures):
                agent_id, topic, result = future.result()
                results[agent_id] = {"subtopic": topic, **result}
                if callback.get("on_subagent_done"):
                    callback["on_subagent_done"](agent_id, result["duration"], topic)

    wall_clock = round(time.time() - start_time, 2)
    serial_sum = round(sum(item["duration"] for item in results.values()), 2)
    stat = {
        "n_subagents": len(results),
        "wall_clock": wall_clock,
        "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall_clock, 2) if wall_clock else 0,
    }
    state["subagents"].update(results)
    state.setdefault("parallel_stats", []).append(stat)

    observations = []
    for agent_id, topic, _ in definitions:
        result = results[agent_id]
        observations.append("【子任务：" + topic + "】\n" + result["final_answer"][:700])
    return "并行子任务完成：" + str(len(results)) + " 个子 Agent；" + str(stat) + "\n\n" + "\n\n".join(observations)


def run_task(question, on_main_step=None, on_subagent_step=None, on_subagent_done=None,
             on_dispatch=None, serial=False):
    """运行一次通用任务，返回最终答案、主 trace、子 trace 和并行统计。"""
    shared_state = {
        "subagents": {},
        "parallel_stats": [],
        "serial": serial,
        "callbacks": {
            "on_subagent_step": on_subagent_step,
            "on_subagent_done": on_subagent_done,
            "on_dispatch": on_dispatch,
        },
    }
    main_agent = ReActLoop(
        "coordinator",
        {
            "web_search": (search_tool, "联网搜索一次，参数是查询词"),
            "dispatch_subagents": (dispatch_subagents, "派发多个子 Agent 并行处理任务，参数用 | 分隔"),
        },
        max_steps=4,
        system_prompt=MAIN_SYSTEM,
    )
    result = main_agent.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
    }
