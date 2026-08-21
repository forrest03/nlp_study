"""
主 Agent + 并行 Subagent 编排【天气经纬度查询版】
最小改动改造：原市场调研 → 批量多城市天气/经纬度查询
教学重点不变：
  1. 主 agent 自身ReAct循环，2个工具：
     - geocode：单城市查经纬度（单一城市简单问题直接用）
     - dispatch_subagents：派发多个subagent并行批量查多城市天气（多城市调研用）
     主LLM自主路由工具，无固定拓扑
  2. 并行优势：ThreadPool多子Agent同时跑，总耗时≈最慢子Agent，非累加
  3. 每个subagent也是独立ReAct，持有geocode+get_weather_by_coords两个工具
     完整trace存入shared_state，支持前端可视化每一步推理过程
架构：Orchestrator-Workers 动态编排（主Agent自主决定拆分多少城市并行）
"""

import os, time, json, logging, uuid, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from weather_tools import geocode, get_weather_by_coords

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是天气调度主分析师。你拥有2个工具：
- geocode：查询单个城市经纬度，参数=城市名称字符串。仅单一城市、只需要坐标时使用
- dispatch_subagents：派发多个子气象调研员并行批量查询多城市天气，参数=管道|分隔的多个城市名称

【关键决策原则】
- 只要问题包含2个及以上城市（批量查询多地天气/坐标对比），
  必须使用 dispatch_subagents 将每个城市拆分为独立子课题并行处理，禁止自身串行多次调用geocode。
  示例："查询广州、惠州、北京3地未来3天天气" → Action: dispatch_subagents
        Action Input: 广州 | 惠州 | 北京
- 仅单一城市需求（如"深圳经纬度"）直接调用 geocode
- 收集全部子城市调研结果后，整合生成结构化天气汇总报告

报告要求：分城市维度整理，每个城市包含坐标+当前天气+3日预报，末尾补充整体总结。

【示例】
Question: 查询上海、成都、西安三地天气
Thought: 需求包含3个城市，属于批量多地点查询，必须派发子Agent并行处理，不串行查询
Action: dispatch_subagents
Action Input: 上海 | 成都 | 西安
Observation: 并行调研完成：3个子调研员...（每个城市完整坐标+天气结果）
Thought: 已收集全部城市数据，整合输出完整汇总天气报告
Final Answer: （分城市结构化汇总报告）"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现
    action_input: "城市1 | 城市2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出需要查询的城市名称"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, city_name) 三元组
    defs = []
    for city in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        def wrapped_geocode(city_name, **_):
            return geocode(city_name)
        def wrapped_get_weather_coords(coord_str, **_):
            # 兼容LLM字符串入参：lat,lon 逗号分割
            lat_str, lon_str = coord_str.split(",")
            return get_weather_by_coords(float(lat_str.strip()), float(lon_str.strip()))

        sub = ReActLoop(
            agent_name=sid,
            tools={
                "geocode": (wrapped_geocode, "输入城市名称，查询该城市经纬度坐标"),
                "get_weather_by_coords": (wrapped_get_weather_coords, "输入「纬度,经度」逗号分隔字符串，查询对应坐标未来3天天气"),
            },
            max_steps=4, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, city))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
    dispatch_info = {"cities": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)   # 真实 id，前端加的节点和后续 subagent_step 对得上

    t0 = time.time()
    results = {}
    # ── 执行：serial=False 并行(ThreadPool) / serial=True 串行(for 循环) ──
    def _run_one(sid=sid, sub=sub, city=city):
        return sid, sub.run(f"查询{city}完整经纬度与未来3天天气", on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
        for sid, sub, city in defs:
            sid, res = _run_one(sid, sub, city)
            target_city = next(t for s, _, t in defs if s == sid)
            results[sid] = (target_city, res)
            shared_state["subagents"][sid] = {
                "city": target_city, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], target_city)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, city): sid for sid, sub, city in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                target_city = next(t for s, _, t in defs if s == sid)
                results[sid] = (target_city, res)
                shared_state["subagents"][sid] = {
                    "city": target_city, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], target_city)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【查询城市: {city}】(用时{r['duration']}s)\n{r['final_answer'][:800]}"
             for sid, (city, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行气象调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次多城市天气批量查询。返回 {final_answer, main_trace, subagents, parallel_stats}。
    serial=True 时 subagent 串行执行（eval A/B 对比基线）。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        # dispatch 事件由 _dispatch_subagents 用真实 subagent id 发出
        # （不能在这里预生成 id，否则和 subagent_step 的 id 对不上）
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    def main_geocode_wrapper(city_name, **_):
        return geocode(city_name)

    main = ReActLoop(
        agent_name="main",
        tools={
            "geocode": (main_geocode_wrapper,
                        "输入城市名，单次查询该城市经纬度坐标，仅单一城市坐标查询使用"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子气象调研员并行批量查询多城市天气，参数=管道|分隔的多个城市名称"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
    )
    # 把 shared_state 注入主 agent run
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
    q = "查询惠州、广州、北京三座城市的经纬度和未来3天天气"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数量: {len(r['subagents'])}")
    print(f"并行性能统计: {r['parallel_stats']}")
    print(f"\n汇总天气报告片段:\n{r['final_answer'][:400]}")

"""
通用 ReAct 循环引擎
教学重点：
  1. ReAct = Reason + Act：LLM 生成 Thought(推理) → Action(选工具) → Action Input(参数)，
     runner 执行工具得 Observation，再喂回 LLM 继续，直到 Final Answer
  2. 主 agent 和 subagent 都是 ReAct 循环——区别只在「有哪些工具」：
     主 agent 有 geocode + dispatch_subagents，subagent 持有 geocode + get_weather_by_coords
  3. 完整 trace 捕获：每步 Thought/Action/ActionInput/Observation 存下来，
     供可视化「点节点看 ReAct 过程」用

用 stop=["Observation:"] 让 LLM 在生成完 Action Input 后停下，runner 执行工具
再补 Observation 续写——这是 ReAct 的经典实现技巧。

依赖：仅 llm_client + 工具函数，无外部库
"""

import time, re, json, logging
from typing import Callable, Optional
from llm_client import llm_chat

logger = logging.getLogger(__name__)

REACT_SYSTEM = """你是气象查询助手，可用工具查询城市坐标与天气。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析需要调用什么工具、入参是什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够坐标与天气信息
Final Answer: 结构化汇总结果，包含坐标、当前天气、3日预报

规则：
- Action 必须是上面列出的工具名之一
- Action Input 严格匹配工具要求格式（城市名 / lat,lon逗号字符串）
- 每轮只调一次工具，等 Observation 返回再决定下一步"""


def build_tools_desc(tools: dict) -> str:
    """把 tools 字典格式化成工具说明。tools: {name: (fn, description)}"""
    lines = []
    for name, (fn, desc) in tools.items():
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


class ReActLoop:
    """通用 ReAct 循环。主 agent / subagent 各自实例化一个。【完全无改动】"""

    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None):
        """
        tools: {tool_name: (fn(arg)->str, description_str)}
        system_prompt: 自定义系统提示（主 agent 用MAIN_SYSTEM引导批量派发）。
                       None 时用默认 REACT_SYSTEM。{tools_desc} 占位符会被替换。
        """
        self.agent_name = agent_name
        self.tools = tools          # {name: (fn, desc)}
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM
        self.trace: list[dict] = []  # 本轮执行 trace（点节点查看用）

    def run(self, question: str, on_step: Callable = None,
            shared_state: dict = None) -> dict:
        """
        执行 ReAct 循环。【函数体完全无改动】
        on_step(step_dict): 每步回调（SSE 流式可视化用）。
        shared_state: 共享状态 dict（主 agent 派发 subagent 时往里塞子Agent trace）。
        返回 {final_answer, trace, duration}。
        """
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        # 对话历史：累积 Thought/Action/ActionInput/Observation
        history = f"Question: {question}\n\n"
        final_answer = ""

        for step_idx in range(self.max_steps):
            # 调 LLM 生成下一步（停在 Observation: 前）
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=768, stop=["Observation:"])
            # 解析 Action 或 Final Answer
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": None}

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input   # Final Answer 内容放 action_input
                self.trace.append(step)
                if on_step: on_step(step)     # final：单次回调
                break

            # ── pre 执行：立即发 step（observation=None），前端实时展示决策 ──
            step["final"] = False
            if on_step: on_step(step)

            # 执行工具（派发子Agent会阻塞等待全部并行完成）
            observation = self._exec_tool(action, action_input, shared_state)

            # ── post 执行：同一 idx 更新为带结果的完整步骤 ──
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step: on_step(step)

            # 续写对话历史
            history += llm_out + f"Observation: {observation[:1200]}\n"

        else:
            # 超过 max_steps，强制收尾
            final_answer = "（已达最大推理步数）" + (self.trace[-1].get("observation","") or "")
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限，整合已有数据输出", "action": "Final Answer",
                    "action_input": final_answer, "observation": None, "final": True}
            self.trace.append(step)
            if on_step: on_step(step)

        duration = round(time.time() - t0, 2)
        return {"final_answer": final_answer, "trace": self.trace,
                "duration": duration}

    def _parse(self, text: str) -> tuple[str, str, str]:
        """LLM输出解析逻辑"""
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m: thought = m.group(1).strip()[:400]

        # Final Answer 优先检测
        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        # Action / Action Input
        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = (mi.group(1).strip() if mi else "")
            return thought, action, action_input

        # 兜底：无格式标记直接输出答案
        if text.strip():
            return thought or "整合坐标天气信息输出报告", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        """工具执行入口"""
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            # 派发工具需要shared_state，普通天气工具忽略
            return str(fn(action_input, shared_state=shared_state)
                       if shared_state is not None else fn(action_input))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


# ReAct单工具自测（天气版本）
if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)

    def wrap_geo(city,**_):
        return geocode(city)
    loop = ReActLoop("test", tools={"geocode": (wrap_geo, "查询城市经纬度")}, max_steps=4)
    r = loop.run("惠州市经纬度是多少？")
    print(f"\n坐标结果: {r['final_answer'][:200]}")
    print(f"推理trace共{len(r['trace'])}步:")
    for s in r["trace"]:
        print(f"  [{s['idx']}] {s['action']}({s['action_input'][:40]}) → {(s.get('observation') or '')[:60]}")

"""极简 LLM 客户端
依赖：pip install openai
"""
import os, time, logging
from openai import OpenAI
logger = logging.getLogger(__name__)
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
_client = None
def get_client():
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key: raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client
def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3):
    """单轮LLM对话，stop截断用于ReAct推理分段"""
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=temperature, max_tokens=max_tokens, stop=stop)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt); logger.warning(f"LLM接口重试({attempt+1}): {str(e)[:80]}")
