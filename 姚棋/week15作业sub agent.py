#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_agent.py - 并行 Subagent 系统（整合版）
功能：
  - 主 Agent 自动拆解多维度问题并并行派发 Subagent
  - 支持 Web 可视化服务（FastAPI + SSE）或命令行运行
  - 内置性能对比（并行 vs 串行）

使用方式：
  1. 命令行运行单个问题：
     python integrated_agent.py --question "中国咖啡市场调研：市场规模、主要品牌、消费趋势"
  2. 运行性能对比（并行 vs 串行）：
     python integrated_agent.py --eval
  3. 启动 Web 服务（需安装 fastapi uvicorn）：
     python integrated_agent.py --serve

环境变量：
  DEEPSEEK_API_KEY  - DeepSeek API Key（必填）
  TAVILY_API_KEY    - Tavily Search API Key（必填）
"""

import os
import sys
import json
import time
import re
import logging
import uuid
import queue
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Dict, List, Any
from urllib import request, error

# 设置日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ======================== 1. Tavily 搜索 ========================
TAVILY_URL = "https://api.tavily.com/search"

def tavily_search(query: str, max_results: int = 5) -> dict:
    """调用 Tavily 搜索 API"""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "未设置 TAVILY_API_KEY"}
    payload = {
        "api_key": key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        req = request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                    "content": (r.get("content") or "")[:600]}
                   for r in data.get("results", [])]
        return {"answer": data.get("answer") or "",
                "results": results,
                "response_time": data.get("response_time")}
    except Exception as e:
        logger.warning(f"Tavily 搜索失败 '{query}': {e}")
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}

def format_search_result(r: dict) -> str:
    if "error" in r:
        return f"搜索失败: {r['error']}"
    parts = []
    if r.get("answer"):
        parts.append(f"摘要: {r['answer']}")
    for i, res in enumerate(r.get("results", []), 1):
        parts.append(f"[{i}] {res['title']}\n    {res['content'][:300]}")
    return "\n".join(parts) if parts else "无结果"

# ======================== 2. LLM 客户端 ========================
from openai import OpenAI  # 需 pip install openai
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
_client = None

def get_client():
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请设置 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client

def llm_chat(system: str, user: str, *, temperature=0.0, max_tokens=1024, stop=None, retries=3) -> str:
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            logger.warning(f"LLM 重试({attempt+1}): {str(e)[:80]}")

# ======================== 3. ReAct 循环引擎 ========================
REACT_SYSTEM_TEMPLATE = """你是市场调研助手，能用以下工具联网搜索调研。

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

class ReActLoop:
    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None):
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM_TEMPLATE
        self.trace = []

    def run(self, question: str, on_step: Callable = None,
            shared_state: dict = None) -> dict:
        self.trace = []
        t0 = time.time()
        tools_desc = "\n".join([f"- {name}: {desc}" for name, (_, desc) in self.tools.items()])
        system = self._system_template.format(tools_desc=tools_desc)
        history = f"Question: {question}\n\n"
        final_answer = ""

        for step_idx in range(self.max_steps):
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=768, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": None}

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            # 执行前推送
            step["final"] = False
            if on_step:
                on_step(step)

            observation = self._exec_tool(action, action_input, shared_state)
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step:
                on_step(step)

            history += llm_out + f"Observation: {observation[:1200]}\n"

        else:
            final_answer = "（已达最大步数）" + (self.trace[-1].get("observation", "") or "")
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": None, "final": True}
            self.trace.append(step)
            if on_step:
                on_step(step)

        duration = round(time.time() - t0, 2)
        return {"final_answer": final_answer, "trace": self.trace,
                "duration": duration}

    def _parse(self, text: str) -> tuple:
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = (mi.group(1).strip() if mi else "")
            return thought, action, action_input

        if text.strip():
            return thought or "综合调研结果给出报告", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            if shared_state is not None:
                return str(fn(action_input, shared_state=shared_state))
            else:
                return str(fn(action_input))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"

# ======================== 4. 主 Agent 与 Subagent 调度 ========================
MAIN_SYSTEM = """你是市场调研主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「市场调研」「竞品分析」「行业分析」「XX 概况/现状/趋势」等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
- 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
- 拿到子调研结果后，综合成结构化报告

报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。"""

def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
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

    def _run_one(sid, sub, topic):
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

def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        return _dispatch_subagents(action_input, shared_state=shared_state,
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
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
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

# ======================== 5. 性能对比工具 ========================
EVAL_QUESTIONS = [
    "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势",
    "中国咖啡市场调研：市场规模、主要品牌、消费趋势",
    "中国扫地机器人市场调研：市场规模、主要品牌、技术趋势",
    "中国宠物经济调研：市场规模、主要品类、消费趋势",
]

def run_eval(limit: int = 0):
    qs = EVAL_QUESTIONS[:limit] if limit else EVAL_QUESTIONS
    results = []
    for i, q in enumerate(qs):
        print(f"[{i+1}/{len(qs)}] {q}")
        # 并行
        p = run_research(q, serial=False)
        # 串行（让主 agent 的 subagent 串行执行，用于对比）
        s = run_research(q, serial=True)
        ps = p["parallel_stats"][-1] if p["parallel_stats"] else {}
        results.append({"question": q,
                        "parallel_wall": ps.get("wall_clock", 0),
                        "serial_wall": ps.get("serial_sum", 0),
                        "speedup": ps.get("speedup", 0)})
        print(f"  并行 {ps.get('wall_clock',0):.2f}s vs 串行 {ps.get('serial_sum',0):.2f}s, 加速 {ps.get('speedup',0):.2f}x")

    if results:
        avg_p = sum(r["parallel_wall"] for r in results) / len(results)
        avg_s = sum(r["serial_wall"] for r in results) / len(results)
        avg_spd = sum(r["speedup"] for r in results) / len(results)
        print("\n" + "="*60)
        print("Parallel vs Serial 对比")
        print("="*60)
        print(f"平均并行墙钟: {avg_p:.2f}s, 平均串行墙钟: {avg_s:.2f}s, 平均加速: {avg_spd:.2f}x")

# ======================== 6. Web 服务（可选） ========================
def run_web_server(port=8002):
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse, FileResponse
        from contextlib import asynccontextmanager
        import uvicorn
    except ImportError:
        print("需要安装 fastapi 和 uvicorn：pip install fastapi uvicorn")
        return

    STATIC_DIR = Path(__file__).parent / "static"
    STATIC_DIR.mkdir(exist_ok=True)
    # 简单写一个 index.html
    (STATIC_DIR / "index.html").write_text("""
    <!DOCTYPE html>
    <html><head><meta charset="UTF-8"><title>Subagent Demo</title></head>
    <body><h1>Subagent 并行调研</h1>
    <input id="q" style="width:60%" placeholder="输入调研问题">
    <button onclick="go()">调研</button>
    <pre id="log" style="white-space:pre-wrap;border:1px solid #ccc;padding:10px;margin-top:10px;max-height:80vh;overflow:auto;"></pre>
    <script>
    function go(){
        const q=document.getElementById('q').value;
        const log=document.getElementById('log');
        log.textContent='';
        const evt=new EventSource('/stream?q='+encodeURIComponent(q));
        evt.onmessage=e=>{
            const d=JSON.parse(e.data);
            if(d.type==='done'){evt.close();return;}
            log.textContent+=JSON.stringify(d,null,2)+'\\n';
        };
    }
    </script>
    </body></html>
    """, encoding="utf-8")

    @asynccontextmanager
    async def lifespan(app):
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/stream")
    async def stream(q: str):
        def event_gen():
            qq = queue.Queue()
            SENTINEL = object()
            def push(ev):
                qq.put(ev)
            def on_main_step(step):
                push({"type": "main_step", **step})
            def on_dispatch(info):
                push({"type": "dispatch", **info})
            def on_subagent_step(sid, step):
                push({"type": "subagent_step", "subagent_id": sid, **step})
            def on_subagent_done(sid, duration, topic):
                push({"type": "subagent_done", "subagent_id": sid,
                      "duration": duration, "subtopic": topic})

            def run():
                try:
                    r = run_research(q,
                                     on_main_step=on_main_step,
                                     on_dispatch=on_dispatch,
                                     on_subagent_step=on_subagent_step,
                                     on_subagent_done=on_subagent_done)
                    push({"type": "final", "answer": r["final_answer"],
                          "parallel_stats": r["parallel_stats"],
                          "subagent_count": len(r["subagents"])})
                except Exception as e:
                    push({"type": "error", "message": str(e)})
                finally:
                    qq.put(SENTINEL)
            threading.Thread(target=run, daemon=True).start()
            while True:
                ev = qq.get()
                if ev is SENTINEL:
                    yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                    break
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
        return StreamingResponse(event_gen(), media_type="text/event-stream")

    print(f"启动 Web 服务: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# ======================== 7. 主入口 ========================
def main():
    parser = argparse.ArgumentParser(description="并行 Subagent 系统")
    parser.add_argument("--question", "-q", help="单次调研问题")
    parser.add_argument("--eval", action="store_true", help="运行性能对比 (并行 vs 串行)")
    parser.add_argument("--serve", action="store_true", help="启动 Web 可视化服务")
    parser.add_argument("--port", type=int, default=8002, help="Web 服务端口")
    parser.add_argument("--limit", type=int, default=0, help="eval 时限制题目数量")
    args = parser.parse_args()

    # 检查必需的环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置环境变量 DEEPSEEK_API_KEY")
        sys.exit(1)
    if not os.getenv("TAVILY_API_KEY"):
        print("错误：请设置环境变量 TAVILY_API_KEY")
        sys.exit(1)

    if args.serve:
        run_web_server(args.port)
    elif args.eval:
        run_eval(args.limit)
    elif args.question:
        print(f"问题: {args.question}\n")
        # 简单打印主 agent 和 subagent 的 trace
        result = run_research(args.question,
                              on_main_step=lambda s: print(f"[主] {s.get('action')} -> {s.get('action_input','')}"),
                              on_dispatch=lambda d: print(f"派发 {d['subagent_ids']}"),
                              on_subagent_step=lambda sid, s: print(f"[{sid}] {s.get('action')}"))
        print("\n最终报告:\n", result["final_answer"][:1000])
        if result["parallel_stats"]:
            stats = result["parallel_stats"][-1]
            print(f"\n并行统计: {stats['n_subagents']} 个 subagent, 墙钟 {stats['wall_clock']}s, 加速 {stats['speedup']}x")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
