"""
主 Agent 编排：支持分发 sub-agent 并行调研

架构：
  主 Agent（ReActLoop）
    ├── 工具: web_search           → 简单搜索，直接查询
    └── 工具: dispatch_subagents   → 拆分子问题，线程池并行派发子 Agent
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           ▼                              ▼                              ▼
    子 Agent 1 (ReActLoop)        子 Agent 2 (ReActLoop)        子 Agent 3 (ReActLoop)
    工具: web_search               工具: web_search               工具: web_search
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

# ── 子 Agent 最大执行时间（秒），防止单个子任务卡死 ──
SUB_AGENT_TIMEOUT = 120

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

MAIN_SYSTEM = """你是资深国产汽车博主／主编，专注中国品牌汽车（比亚迪、蔚来、小鹏、理想、吉利、长城、小米等）的深度评测与对比分析，能调用以下工具。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析任务结构，判断是否需要拆分子问题（按品牌、车型、维度等拆分）
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（用博主口吻，带来源要点，适合发公众号/小红书）

规则：
- 如果问题涉及多个品牌对比、多款车型对比、或多个维度分析（如外观/性能/价格/续航），用 dispatch_subagents 拆成子问题并行调研
- dispatch_subagents 的 Action Input 用 "||" 分隔多个子问题，例如：比亚迪汉2024款性能参数 || 蔚来ET7 2024款性能参数 || 小鹏P7 2024款性能参数
- 如果问题简单、只需一个查询，直接用 web_search
- 收到子调研结果后，综合所有信息给出 Final Answer
- 每轮只调一次工具，等 Observation 再决定下一步"""

SUB_SYSTEM = """你是汽车行业研究员，专注中国品牌汽车的参数、价格、口碑等信息的搜集与整理。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（带来源要点）

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步"""

# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════


def _web_search(query: str, **_kwargs) -> str:
    """联网搜索工具"""
    return format_search_result(tavily_search(query))


def dispatch_subagents(action_input: str, shared_state: dict = None) -> str:
    """派发子 Agent 工具：用 || 分隔多个子问题，线程池并行执行。

    action_input 示例: "比亚迪汉2024款性能参数 || 蔚来ET7 2024款性能参数"

    每个子问题创建一个 ReActLoop 子 Agent（仅 web_search 工具），
    通过 ThreadPoolExecutor 并行执行，全部完成后汇总结果返还。
    """
    if shared_state is None:
        shared_state = {}

    # 解析子问题（用 || 分隔）
    sub_questions = [q.strip() for q in action_input.split("||") if q.strip()]
    if not sub_questions:
        return "dispatch_subagents 错误: 未提供有效的子问题，请用 || 分隔"

    logger.info(f"派发 {len(sub_questions)} 个子 Agent: {sub_questions}")

    # 子 Agent 共享的工具集：只有 web_search
    sub_tools = {
        "web_search": (_web_search, "联网搜索，参数是查询词"),
    }

    # 为每个子问题创建 ReActLoop 实例
    sub_agents = []
    for i, q in enumerate(sub_questions):
        name = f"子研究员-{i + 1}"
        sub_loop = ReActLoop(
            agent_name=name,
            tools=sub_tools,
            max_steps=3,                     # 子 Agent 步数限制
            system_prompt=SUB_SYSTEM,
        )
        sub_agents.append((sub_loop, q))

    # 初始化 shared_state 中的 subagent_traces（如果还没有）
    if "subagent_traces" not in shared_state:
        shared_state["subagent_traces"] = []

    # ── 线程池并行执行 ──
    results = []
    with ThreadPoolExecutor(max_workers=len(sub_agents)) as executor:
        future_map = {}
        for sub_loop, q in sub_agents:
            future = executor.submit(sub_loop.run, q, shared_state=shared_state)
            future_map[future] = (sub_loop.agent_name, q)

        for future in as_completed(future_map):
            name, question = future_map[future]
            try:
                result = future.result(timeout=SUB_AGENT_TIMEOUT)
                # 将子 Agent 的 trace 存入 shared_state
                shared_state["subagent_traces"].append({
                    "agent_name": name,
                    "question": question,
                    "trace": result["trace"],
                    "duration": result["duration"],
                })
                results.append({
                    "agent_name": name,
                    "question": question,
                    "final_answer": result["final_answer"],
                    "duration": result["duration"],
                })
            except TimeoutError:
                logger.warning(f"{name} 超时（{SUB_AGENT_TIMEOUT}s）")
                results.append({
                    "agent_name": name,
                    "question": question,
                    "final_answer": f"（超时，{SUB_AGENT_TIMEOUT}s 内未完成）",
                    "duration": SUB_AGENT_TIMEOUT,
                })
            except Exception as e:
                logger.error(f"{name} 执行异常: {e}")
                results.append({
                    "agent_name": name,
                    "question": question,
                    "final_answer": f"（执行异常: {str(e)[:120]}）",
                    "duration": 0,
                })

    # ── 格式化汇总结果，返回给主 Agent ──
    output_parts = []
    for r in results:
        output_parts.append(
            f"[{r['agent_name']}]\n"
            f"子问题: {r['question']}\n"
            f"调研结果: {r['final_answer'][:600]}\n"
            f"耗时: {r['duration']}s"
        )
    return "\n\n---\n\n".join(output_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 主 Agent 工厂
# ═══════════════════════════════════════════════════════════════════════════════

def create_main_agent(model_tag: str = "deepseek-v4-flash") -> ReActLoop:
    """创建主 Agent 实例，配备 web_search 和 dispatch_subagents 两个工具。"""
    main_tools = {
        "web_search": (_web_search, "联网搜索汽车信息，参数是查询词。适合单一车型/品牌快速查询"),
        "dispatch_subagents": (
            dispatch_subagents,
            "派发子研究员并行调研多款车型或品牌。参数用 || 分隔多个子问题，"
            "例如：比亚迪汉2024款性能参数 || 蔚来ET7 2024款性能参数。子研究员各自搜索后汇总结果",
        ),
    }
    return ReActLoop(
        agent_name="汽车主编",
        tools=main_tools,
        max_steps=6,                      # 主 Agent 步数比子 Agent 多
        model_tag=model_tag,
        system_prompt=MAIN_SYSTEM,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    main_agent = create_main_agent()
    shared_state = {}  # 共享状态，dispatch_subagents 往里写子 Agent trace

    question = "对比分析比亚迪汉2026款和小米SU7 2026款的性能、价格和续航"
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"{'='*60}")

    result = main_agent.run(question, shared_state=shared_state)

    print(f"\n{'='*60}")
    print(f"最终答案:")
    print(f"{'='*60}")
    print(result["final_answer"])
    print(f"\n总耗时: {result['duration']}s")

    # 打印主 Agent trace 概览
    print(f"\n{'='*60}")
    print(f"主 Agent Trace ({len(result['trace'])} 步):")
    print(f"{'='*60}")
    for s in result["trace"]:
        action = s.get("action", "")
        if action == "Final Answer":
            print(f"  [{s['idx']}] Final Answer: {(s.get('action_input') or '')[:100]}...")
        elif action:
            obs = (s.get("observation") or "")[:80]
            print(f"  [{s['idx']}] {action} → {obs}")

    # 打印子 Agent trace 概览
    sub_traces = shared_state.get("subagent_traces", [])
    if sub_traces:
        print(f"\n{'='*60}")
        print(f"子 Agent Traces ({len(sub_traces)} 个):")
        print(f"{'='*60}")
        for st in sub_traces:
            print(f"  {st['agent_name']}: {st['question'][:40]}... ({st['duration']}s, {len(st['trace'])} 步)")