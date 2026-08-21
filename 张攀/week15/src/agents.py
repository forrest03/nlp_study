"""
主 Agent + 并行 Subagent 编排

教学重点：
  1. 主 agent 自己是 ReAct 循环，有 4 个工具：
     - web_search：单次联网搜索（简单问题直接用）
     - dispatch_subagents：派发多个 subagent 并行调研（多侧面研究问题用）
     - geocode：地理编码
     - get_weather_by_coords：根据坐标获取天气（天气类问题专用）
     主 agent 根据 query 自行决定用哪个——不是固定拓扑，是 LLM 自主路由
  2. 并行优势凸显：dispatch_subagents 一次派发 N 个 subagent，
     ThreadPoolExecutor 并行跑，wall-clock ≈ max(单agent时长)，
     而非 sum——这就是 subagent 并行的核心价值
  3. 每个 subagent 也是 ReAct 循环（包含web_search、geocode、get_weather_by_coords 工具），
     trace 全程捕获存入 shared_state，供可视化「点节点看 ReAct 过程」

架构对应 PPT Part 6.3 的 Orchestrator-Workers 拓扑（动态：主 agent 决定派几个）。
"""

import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

# MAIN_SYSTEM = """你是市场调研主分析师。你有 2 个工具：
# - web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
# - dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

# 【关键决策原则】
# - 只要问题涉及 2 个及以上侧面（如「市场调研」「竞品分析」「行业分析」「XX 概况/现状/趋势」等），
#   必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
#   示例："新能源汽车市场调研：销量、竞争、政策" → Action: dispatch_subagents
#         Action Input: 2024年中国新能源汽车销量规模 | 主要厂商竞争格局 | 政策与补贴趋势
# - 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
# - 拿到子调研结果后，综合成结构化报告

# 报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。

# 【示例】
# Question: 2023中国咖啡市场调研：市场规模、主要品牌、消费趋势
# Thought: 这是多维度市场调研（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
# Action: dispatch_subagents
# Action Input: 2023年中国咖啡市场规模与增长 | 中国咖啡主要品牌竞争格局 | 中国咖啡消费趋势与人群
# Observation: 并行调研完成：3 个子调研员...（各子课题结果）
# Thought: 已收齐三个维度的并行调研结果，综合成报告
# Final Answer: （分维度报告）"""

# MAIN_SYSTEM = """
# 你是全能主分析师。你有 4 个工具：
# - web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
# - dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）
# - geocode：根据地区名称获取经纬度（参数=地区名称）
# - get_weather_by_coords：根据经纬度获取天气（参数=用 , 分隔的纬度和经度）

# 【关键决策原则】
# - 只要问题涉及 2 个及以上侧面（如「调研」「分析」「XX 概况/现状/趋势/天气」等），
#   必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
#   示例："新能源汽车市场调研：销量、竞争、政策" → Action: dispatch_subagents
#         Action Input: 2024年中国新能源汽车销量规模 | 主要厂商竞争格局 | 政策与补贴趋势
#         香格里拉旅游攻略：天气、景点、美食 → Action: dispatch_subagents
#         Action Input: 香格里拉未来天气 | 香格里拉必游景点 | 香格里拉美食推荐
# - 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
# - 单一事实问题如果是关于天气，必须先用 geocode 获取坐标，再用 get_weather_by_coords 获取天气，不能 web_search
# - 拿到子调研结果后，综合成结构化报告

# 报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。

# 【示例】
# Question: 2023中国咖啡市场调研：市场规模、主要品牌、消费趋势
# Thought: 这是多维度市场调研（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
# Action: dispatch_subagents
# Action Input: 2023年中国咖啡市场规模与增长 | 中国咖啡主要品牌竞争格局 | 中国咖啡消费趋势与人群
# Observation: 并行调研完成：3 个子调研员...（各子课题结果）
# Thought: 已收齐三个维度的并行调研结果，综合成报告
# Final Answer: （分维度报告）

# Question: 香格里拉旅游攻略：天气、景点、美食
# Thought: 这是多维度旅游攻略（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
# Action: dispatch_subagents
# Action Input: 香格里拉未来天气 | 香格里拉必游景点 | 香格里拉美食推荐
# Observation: 并行调研完成：3 个子调研员...（各子课题结果）
# Thought: 已收齐三个维度的并行调研结果，综合成报告
# Final Answer: （分维度报告）
# """

MAIN_SYSTEM = """
你是全能主分析师。你有 4 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）
- geocode：根据地区名称获取经纬度（参数=地区名称）
- get_weather_by_coords：根据经纬度获取天气（参数=用 , 分隔的纬度和经度）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「调研」「分析」「XX 概况/现状/趋势/天气」等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
  示例："新能源汽车市场调研：销量、竞争、政策" → Action: dispatch_subagents
        Action Input: 2024年中国新能源汽车销量规模 | 主要厂商竞争格局 | 政策与补贴趋势
        香格里拉旅游攻略：天气、景点、美食 → Action: dispatch_subagents
        Action Input: 香格里拉未来天气 | 香格里拉必游景点 | 香格里拉美食推荐
- 只有单一事实问题（如"2024年比亚迪销量"）可直接 web_search，如果单一事实是关于天气，把这个问题作为子课题，用 dispatch_subagents 派发子调研员
- 拿到子调研结果后，综合成结构化报告

报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。

【示例】
Question: 2023中国咖啡市场调研：市场规模、主要品牌、消费趋势
Thought: 这是多维度市场调研（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 2023年中国咖啡市场规模与增长 | 中国咖啡主要品牌竞争格局 | 中国咖啡消费趋势与人群
Observation: 并行调研完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三个维度的并行调研结果，综合成报告
Final Answer: （分维度报告）

Question: 香格里拉旅游攻略：天气、景点、美食
Thought: 这是多维度旅游攻略（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 香格里拉未来天气 | 香格里拉必游景点 | 香格里拉美食推荐
Observation: 并行调研完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三个维度的并行调研结果，综合成报告
Final Answer: （分维度报告）
"""

import httpx
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}

def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, subtopic) 三元组
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={
                "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                               "联网搜索，参数是查询词"),
                "geocode": (geocode,
                            "根据城市名查经纬度，参数=城市名"),
                "get_weather_by_coords": (get_weather_by_coords,
                                          "根据经纬度查天气，参数=用 , 分隔的纬度和经度"),

            },
            max_steps=4, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, topic))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)   # 真实 id，前端加的节点和后续 subagent_step 对得上

    t0 = time.time()
    results = {}
    # ── 执行：serial=False 并行(ThreadPool) / serial=True 串行(for 循环) ──
    def _run_one(sid=sid, sub=sub, topic=topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
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
        # 并行（凸显 subagent 并行优势的核心）
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

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))

def geocode(city: str) -> str:
    """
    工具一：城市名 → 经纬度（Geocoding 接口）。

    返回文字描述，里面明确包含 latitude / longitude，方便模型把这两个数
    传给下一个工具 get_weather_by_coords 完成链式调用，也能独立回答"X 的经纬度"。
    """
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(GEOCODING_URL, params={
            "name": city, "count": 10, "language": "zh", "format": "json",
        })
        resp.raise_for_status()
        results = resp.json().get("results") or []

        # 与原 backend 同样的同名小村庄消歧策略：裸低级行政点且没带"市/县/区"后缀，
        # 就用 city+"市" 重查一次并优先采用。
        def _geocode(name: str):
            r = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            r.raise_for_status()
            return r.json().get("results") or []

        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            return (admin_priority, r.get("population") or 0)

        loc = max(results, key=_rank)
        lat = loc["latitude"]
        lon = loc["longitude"]
        location_str = f"{loc.get('country', '')} {loc.get('admin1', '')} {loc.get('name', city)}".strip()
        return (
            f"城市：{location_str}\n"
            f"纬度(latitude)：{lat}\n"
            f"经度(longitude)：{lon}"
        )

# def get_weather_by_coords(latitude: float, longitude: float) -> str:
def get_weather_by_coords(location: str) -> str:
    """
    工具二：经纬度 → 天气（Forecast 接口）。

    只要拿到经纬度就能直接查，不需要城市名。所以用户直接给经纬度也能答，
    模型链式调用时把 geocode 的输出喂进来即可。
    """
    # location为用逗号分隔的维度和经度字符串，如"??39.9042???,???116.4074???"
    # 用正则表达式剔除无关字符，保留数字、点号和逗号
    import re
    location = re.sub(r"[^0-9.,]", "", location)
    latitude, longitude = map(float, location.split(","))

    with httpx.Client(timeout=10.0) as client:
        try:
            resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = resp.json()
        cur = data["current"]
        daily = data["daily"]
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")

        lines = [
            f"坐标：{latitude}°N, {longitude}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )
        return "\n".join(lines)

def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次市场调研。返回 {final_answer, main_trace, subagents, parallel_stats}。
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

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)), 
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool, 
                                   "派发多个子调研员并行调研，参数=用 | 分隔的多个子课题"),
            "geocode": (lambda q, **_: geocode(q),
                        "根据城市名查经纬度，参数=城市名"),
            "get_weather_by_coords": (lambda q, **_: get_weather_by_coords(q),
                                      "根据经纬度查天气，参数=用 , 分隔的纬度和经度"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,   # ← 传主 agent 的派发引导 prompt
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
    # q = "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势"
    q = "北京旅游攻略：天气、景点、美食"
    # q = "上海天气"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:200]}")
