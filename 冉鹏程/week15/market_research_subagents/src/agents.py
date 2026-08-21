"""企业信息调查 Agent 的主从编排。

主 Agent 负责拆解企业尽调任务并综合结论；子 Agent 并行核验企业基本信息、
规模业务、经营财务、薪酬福利和风险事件。完整 ReAct trace 会保存在返回值中，
供 SSE 与前端拓扑查看。
"""

import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from llm_client import llm_chat
from tavily_search import format_search_result, tavily_search

logger = logging.getLogger(__name__)

MAX_SUBAGENTS = 6
SINGLE_FACT_PATTERN = re.compile(r"多少|几[个位名家条项]|是什么|谁是|何时|哪年|总部.{0,4}(?:在哪|哪里)|"
                                 r"成立时间|是否上市|股票代码")
REQUIRED_DIMENSIONS = (
    "企业基本信息：工商主体、成立时间、总部、所有制与融资/上市状态",
    "企业规模与主营业务：员工规模、业务线、产品服务、市场覆盖与核心客户",
    "财务与运营状况：营收、利润、现金流、融资、订单、门店/用户等可验证经营指标",
    "发展前景与岗位稳定性：行业位置、增长计划、招聘动态、组织调整与裁员风险",
    "薪酬福利与工作体验：公开岗位薪资、社保公积金、假期、奖金、培训及员工评价",
    "风险与负面信息：司法案件、行政处罚、失信/执行、劳资争议与权威媒体负面报道",
)

MAIN_SYSTEM = """你是企业信息调查主分析师，为求职者提供审慎、可追溯的择业参考。
你有两个工具：
- web_search：联网搜索一次（参数为查询词），仅用于用户只问一个可核验事实时。
- dispatch_subagents：派发多个子调研员并行调查（参数用 | 分隔）。

【强制编排规则】
- 只要用户请求公司调查、求职参考、公司评价或推荐值，必须调用一次
  dispatch_subagents，并覆盖六项：基本信息、规模与业务、财务/运营、发展与岗位稳定性、
  薪酬福利、风险/负面信息。每个子课题必须写清企业名称，使用 | 分隔。
- 不要自己连续调用 web_search 来代替并行调查；单一事实问题才可直接搜索。
- 拿到子调研结果后，只综合已给出的证据，不得虚构财务数据、薪资或负面事实。

【证据与风险规则】
- 优先公司官网/年报/招股书、交易所公告、国家企业信用信息公示系统、法院/监管机关、
  招聘官网和可信媒体。每条关键结论都标注来源编号与 URL；说明信息截至日期。
- 非上市或未披露企业的营收、利润、现金流必须写“公开信息未披露”，可用融资、招聘、
  门店、用户等代理指标，但要标注为代理指标和局限。
- 负面信息必须说明来源、时间、事件状态与影响。仅有传闻、投诉或尚未证实的报道必须
  标注“待核验”，不得写成事实；不得使用侮辱、诽谤或绝对化措辞。

【最终报告格式】
# 企业求职尽调报告
## 调查范围与信息时点
## 1. 基本信息与规模
## 2. 主营业务与行业位置
## 3. 财务/运营与发展趋势
## 4. 薪酬福利与工作体验
## 5. 风险与负面信息核验
## 6. 求职推荐值（0–100）
用表格列出：经营稳健性 30 分、业务与成长性 20 分、薪酬福利 20 分、
雇主体验 15 分、合规与风险 15 分；每项写证据、得分和信息充分度。信息不足时保守
给分并明确“无法判断”，不要用猜测补齐。总分分档：80–100 值得优先考虑；60–79 可考虑；
40–59 谨慎；0–39 不建议优先。该分数只代表公开信息下的企业就业吸引力，不等同于对
具体岗位、直属团队或个人匹配度的保证。
## 7. 给求职者的建议
列出适合关注的人群、面试时应核实的问题与尚待验证的信息。
## 来源清单
按【S1】格式列出来源名称、URL 与访问/披露日期。

【示例】
Question: 调查某公司，给 Java 后端求职者做参考
Thought: 这是完整企业尽调，必须并行覆盖六个维度
Action: dispatch_subagents
Action Input: 某公司工商主体、成立时间与融资上市状态 | 某公司员工规模、主营业务与市场覆盖 | 某公司营收利润融资与经营指标 | 某公司行业前景、招聘动态与岗位稳定性 | 某公司公开薪资福利与员工工作体验 | 某公司司法行政风险与权威媒体负面信息
Observation: 并行核验完成……
Thought: 已收齐不同维度的证据，按证据强度和信息缺口输出审慎报告
Final Answer: # 企业求职尽调报告……"""

SUBAGENT_SYSTEM = """你是企业信息调查子调研员，负责核验一个明确维度。

可用工具：
{tools_desc}

按如下格式严格输出：
Thought: 说明还需要核验什么
Action: 工具名
Action Input: 查询词

工具执行后会返回 Observation。完成后使用：
Thought: 已完成该维度核验
Final Answer: 仅输出本维度的事实、信息缺口和来源证据（每条含 URL）。

规则：优先官方、监管、法院、交易所、招聘官网及可信媒体；财务或薪资未公开时明确写
“公开信息未披露”。风险信息要区分已生效处罚/判决、正在进行的案件与待核验报道，不能
把传闻当事实。不要计算总推荐值，也不要补造没有搜索到的数据。"""


def build_company_subtopics(question: str, action_input: str) -> list[str]:
    """构造完整的企业尽调子课题。

    参数：question 为用户原始调查请求；action_input 为主 Agent 提议的管道分隔子课题。
    返回：最多六个、可直接检索的子课题；提议不完整时回退为六个必查维度。
    异常：question 为空或超过长度限制时抛出 ValueError。
    """
    normalized_question = _validate_question(question)
    proposed = _split_subtopics(action_input)
    if len(proposed) >= len(REQUIRED_DIMENSIONS):
        return proposed[:MAX_SUBAGENTS]
    return [_build_dimension_query(normalized_question, dimension)
            for dimension in REQUIRED_DIMENSIONS]


def _validate_question(question: str) -> str:
    """清理并校验用户调查请求，避免超长或空输入进入模型和检索服务。"""
    if not isinstance(question, str):
        raise ValueError("调查请求必须是文本")
    normalized = " ".join(question.split())
    if not 2 <= len(normalized) <= 240:
        raise ValueError("调查请求长度须为 2 至 240 个字符")
    return normalized


def _split_subtopics(action_input: str) -> list[str]:
    """去重并限制主 Agent 传入的子课题数量。"""
    items = []
    for topic in str(action_input or "").split("|"):
        normalized = " ".join(topic.split())
        if normalized and normalized not in items:
            items.append(normalized[:300])
    return items[:MAX_SUBAGENTS]


def _build_dimension_query(question: str, dimension: str) -> str:
    """将原始请求与必查维度组合为 Tavily 可接受的查询长度。"""
    delimiter = "｜"
    available = 300 - len(dimension) - len(delimiter)
    return f"{question[:available]}{delimiter}{dimension}"


def _web_search(query: str, shared_state: dict = None) -> str:
    """执行带关联标识的搜索并格式化证据，供主从 Agent 共用。"""
    request_id = (shared_state or {}).get("request_id")
    result = tavily_search(query, request_id=request_id)
    return format_search_result(result)


def _create_subagent(agent_id: str) -> ReActLoop:
    """创建仅拥有联网核验能力的企业调查子 Agent。"""
    return ReActLoop(
        agent_name=agent_id,
        tools={"web_search": (_web_search, "联网检索企业公开证据，参数是查询词")},
        max_steps=4,
        model_tag="deepseek-chat(企业核验子 Agent)",
        system_prompt=SUBAGENT_SYSTEM,
    )


def _run_subagent(agent_id: str, subagent: ReActLoop, topic: str,
                  request_id: str, on_subagent_step: Callable = None) -> tuple[str, dict]:
    """运行一个子 Agent；失败时保留失败说明，让其他维度仍可完成。"""
    started_at = time.time()
    try:
        result = subagent.run(
            topic,
            on_step=lambda step: on_subagent_step(agent_id, step) if on_subagent_step else None,
            shared_state={"request_id": request_id},
        )
        return agent_id, result
    except Exception as error:
        duration = round(time.time() - started_at, 2)
        logger.exception("company_subagent_failed request_id=%s agent_id=%s", request_id, agent_id)
        message = f"该维度暂未完成核验：{type(error).__name__}，请稍后重试或人工补充来源。"
        return agent_id, {"final_answer": message, "duration": duration, "trace": []}


def _store_subagent_result(shared_state: dict, agent_id: str, topic: str, result: dict,
                           on_subagent_done: Callable = None) -> None:
    """保存子 Agent 结果，并在任务完成后通知流式调用方。"""
    shared_state["subagents"][agent_id] = {
        "subtopic": topic,
        "trace": result["trace"],
        "duration": result["duration"],
        "final_answer": result["final_answer"],
    }
    if on_subagent_done:
        on_subagent_done(agent_id, result["duration"], topic)


def _execute_subagents(definitions: list[tuple[str, ReActLoop, str]], shared_state: dict,
                       on_subagent_step: Callable, on_subagent_done: Callable,
                       serial: bool) -> dict[str, tuple[str, dict]]:
    """按串行或并行方式执行子 Agent，并收集每个维度的结果。"""
    results = {}
    request_id = shared_state["request_id"]
    topic_by_id = {agent_id: topic for agent_id, _, topic in definitions}

    def collect(agent_id: str, result: dict) -> None:
        topic = topic_by_id[agent_id]
        results[agent_id] = (topic, result)
        _store_subagent_result(shared_state, agent_id, topic, result, on_subagent_done)

    if serial:
        for agent_id, subagent, topic in definitions:
            completed_id, result = _run_subagent(agent_id, subagent, topic, request_id, on_subagent_step)
            collect(completed_id, result)
        return results

    with ThreadPoolExecutor(max_workers=len(definitions)) as pool:
        futures = [pool.submit(_run_subagent, agent_id, subagent, topic, request_id, on_subagent_step)
                   for agent_id, subagent, topic in definitions]
        for future in as_completed(futures):
            completed_id, result = future.result()
            collect(completed_id, result)
    return results


def _build_dispatch_observation(results: dict[str, tuple[str, dict]], stats: dict) -> str:
    """把子 Agent 的精简证据和并行统计组合为主 Agent 的 Observation。"""
    summaries = []
    for _, (topic, result) in results.items():
        summaries.append(f"【子课题：{topic}】（用时 {result['duration']}s）\n"
                         f"{result['final_answer'][:700]}")
    header = (f"并行核验完成：{stats['n_subagents']} 个子调研员，wall-clock {stats['wall_clock']}s "
              f"（串行预计 {stats['serial_sum']}s，加速 {stats['speedup']}×）")
    return f"{header}\n\n" + "\n\n".join(summaries)


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None, on_subagent_done: Callable = None,
                        on_dispatch: Callable = None, serial: bool = False) -> str:
    """派发六个企业尽调子 Agent，并返回供主 Agent 综合的证据摘要。"""
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("request_id", uuid.uuid4().hex)
    shared_state.setdefault("question", "企业信息调查")
    shared_state.setdefault("subagents", {})
    subtopics = build_company_subtopics(shared_state["question"], action_input)
    definitions = [(f"sub_{uuid.uuid4().hex[:6]}", None, topic) for topic in subtopics]
    definitions = [(agent_id, _create_subagent(agent_id), topic) for agent_id, _, topic in definitions]
    dispatch_info = {"subtopics": subtopics, "subagent_ids": [item[0] for item in definitions]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    logger.info("company_subagents_dispatched request_id=%s count=%s serial=%s",
                shared_state.get("request_id"), len(definitions), serial)
    if on_dispatch:
        on_dispatch(dispatch_info)
    started_at = time.time()
    results = _execute_subagents(definitions, shared_state, on_subagent_step, on_subagent_done, serial)
    wall_clock = round(time.time() - started_at, 2)
    serial_sum = round(sum(result["duration"] for _, result in results.values()), 2)
    stats = {"n_subagents": len(definitions), "wall_clock": wall_clock, "serial_sum": serial_sum,
             "speedup": round(serial_sum / wall_clock, 2) if wall_clock else 0}
    shared_state.setdefault("parallel_stats", []).append(stats)
    logger.info("company_subagents_completed request_id=%s count=%s wall_clock=%s",
                shared_state.get("request_id"), len(definitions), wall_clock)
    return _build_dispatch_observation(results, stats)


def _requires_full_company_survey(question: str) -> bool:
    """判断请求是否需强制覆盖六维企业调查，而非只查询一个事实。"""
    full_survey_terms = ("调查", "尽调", "求职", "推荐", "公司评价", "企业评价", "福利", "薪资", "风险", "负面")
    return any(term in question for term in full_survey_terms) or not SINGLE_FACT_PATTERN.search(question)


def _run_required_dispatch(question: str, shared_state: dict,
                           on_main_step: Callable, on_subagent_step: Callable,
                           on_subagent_done: Callable, on_dispatch: Callable,
                           serial: bool) -> tuple[list[dict], str]:
    """记录并执行企业调查的强制派发步骤，确保前端和报告都有完整六维证据。"""
    action_input = " | ".join(_build_dimension_query(question, dimension)
                               for dimension in REQUIRED_DIMENSIONS)
    dispatch_step = {"idx": 0, "agent": "main", "thought": "企业求职调查必须并行核验六个维度",
                     "action": "dispatch_subagents", "action_input": action_input,
                     "observation": None, "final": False}
    if on_main_step:
        on_main_step(dispatch_step)
    observation = _dispatch_subagents(
        action_input, shared_state, on_subagent_step, on_subagent_done, on_dispatch, serial)
    dispatch_step["observation"] = observation
    dispatch_step["done"] = True
    if on_main_step:
        on_main_step(dispatch_step)
    return [dispatch_step], observation


def _extract_final_answer(model_output: str) -> str:
    """提取模型的最终报告正文，兼容模型省略 ReAct 前缀的情况。"""
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.S)
    return (match.group(1) if match else model_output).strip()


def _synthesize_company_report(question: str, observation: str, request_id: str) -> str:
    """在六维核验完成后调用主模型汇总，不允许再次派发或检索。"""
    synthesis_prompt = (MAIN_SYSTEM + "\n\n【当前执行状态】六个子调研员已经完成核验。"
                        "现在只能直接输出最终报告，不得输出 Action、不得再次检索或派发。")
    history = f"Question: {question}\n\nObservation: {observation}\n\nFinal Answer:"
    try:
        return _extract_final_answer(llm_chat(synthesis_prompt, history, max_tokens=1800))
    except Exception as error:
        logger.exception("company_report_synthesis_failed request_id=%s", request_id)
        return "企业公开信息核验已完成，但报告汇总暂时失败。请根据以下分维度证据人工核验。\n\n" + observation


def _run_full_company_survey(question: str, shared_state: dict,
                             on_main_step: Callable, on_subagent_step: Callable,
                             on_subagent_done: Callable, on_dispatch: Callable,
                             serial: bool) -> dict:
    """执行六维强制派发和报告汇总，返回主 Agent trace 与最终报告。"""
    main_trace, observation = _run_required_dispatch(
        question, shared_state, on_main_step, on_subagent_step, on_subagent_done, on_dispatch, serial)
    final_answer = _synthesize_company_report(question, observation, shared_state["request_id"])
    final_step = {"idx": 1, "agent": "main", "thought": "已汇总六维公开证据与信息缺口",
                  "action": "Final Answer", "action_input": final_answer,
                  "observation": None, "final": True}
    main_trace.append(final_step)
    if on_main_step:
        on_main_step(final_step)
    return {"final_answer": final_answer, "main_trace": main_trace}


def _run_single_fact_research(question: str, shared_state: dict,
                              on_main_step: Callable, on_subagent_step: Callable,
                              on_subagent_done: Callable, on_dispatch: Callable,
                              serial: bool) -> dict:
    """保留单一事实请求的 ReAct 路由，避免不必要地启动六个子 Agent。"""
    def dispatch_tool(action_input: str, shared_state: dict = None) -> str:
        return _dispatch_subagents(action_input, shared_state or {}, on_subagent_step,
                                   on_subagent_done, on_dispatch, serial)

    main = ReActLoop(
        agent_name="main",
        tools={"web_search": (_web_search, "联网检索单一企业事实，参数是查询词"),
               "dispatch_subagents": (dispatch_tool, "派发企业尽调子 Agent，参数用 | 分隔")},
        max_steps=8,
        model_tag="deepseek-chat(企业调查主 Agent)",
        system_prompt=MAIN_SYSTEM,
        observation_limit=4800,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {"final_answer": result["final_answer"], "main_trace": result["trace"]}


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None, on_subagent_done: Callable = None,
                 on_dispatch: Callable = None, serial: bool = False,
                 request_id: str = None) -> dict:
    """执行企业信息调查。

    参数：question 为企业名称及可选意向岗位/城市；各回调用于流式展示；serial 仅用于 A/B
    基准；request_id 用于日志追踪。返回最终报告、主从 trace、派发记录和并行统计。
    异常：输入不合法时抛出 ValueError，调用方应向用户返回校验错误。
    """
    normalized_question = _validate_question(question)
    research_id = request_id or uuid.uuid4().hex
    shared_state = {"request_id": research_id, "question": normalized_question, "subagents": {},
                    "dispatches": [], "parallel_stats": []}
    logger.info("company_research_started request_id=%s question_length=%s", research_id, len(normalized_question))

    if _requires_full_company_survey(normalized_question):
        result = _run_full_company_survey(
            normalized_question, shared_state, on_main_step, on_subagent_step,
            on_subagent_done, on_dispatch, serial)
    else:
        result = _run_single_fact_research(
            normalized_question, shared_state, on_main_step, on_subagent_step,
            on_subagent_done, on_dispatch, serial)
    logger.info("company_research_completed request_id=%s main_steps=%s subagents=%s",
                research_id, len(result["main_trace"]), len(shared_state["subagents"]))
    return {"final_answer": result["final_answer"], "main_trace": result["main_trace"],
            "subagents": shared_state["subagents"], "parallel_stats": shared_state["parallel_stats"],
            "dispatches": shared_state["dispatches"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_question = "调查小米集团，为产品经理求职者提供企业信息、福利和风险参考"
    report = run_research(demo_question)
    print(f"\n主 Agent 动作：{[step['action'] for step in report['main_trace']]}")
    print(f"派发次数：{len(report['dispatches'])}｜子 Agent：{len(report['subagents'])}")
    print(f"并行统计：{report['parallel_stats']}")
    print(f"\n报告：\n{report['final_answer'][:600]}")
