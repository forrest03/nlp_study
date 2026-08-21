"""
display — 结果展示层（线程安全）
================================

subagent 并行执行时多个线程会同时打印，因此所有打印都经过统一锁。

展示内容：
  1. 编排者主循环轨迹（Action / Observation / 最终汇总）
  2. subagent 实时状态（启动 / 完成），并行时可看到交错推进
  3. 每次下发的完整报告：每个 subagent 一个分栏块（目标 / 统计 / 工具调用 / 最终结果）
  4. LLM 用量统计
"""
from __future__ import annotations

import json
import threading

_LOCK = threading.Lock()

COLORS = {
    "cyan":    "\033[36m",
    "yellow":  "\033[33m",
    "green":   "\033[32m",
    "magenta": "\033[35m",
    "red":     "\033[31m",
    "blue":    "\033[34m",
    "bold":    "\033[1m",
    "dim":     "\033[2m",
    "reset":   "\033[0m",
}

_RULE = "═" * 74


def _c(text: str, *colors: str) -> str:
    codes = "".join(COLORS[c] for c in colors)
    return f"{codes}{text}{COLORS['reset']}"


def _trunc(text: str, n: int = 500) -> str:
    text = text.replace("\r", "")
    return text if len(text) <= n else text[:n] + f"…（已截断，共 {len(text)} 字符）"


def _block(title: str, body: str, color: str = "cyan") -> None:
    """一个标题 + 内容的展示块（线程安全）。"""
    with _LOCK:
        print()
        print(_RULE)
        print(_c(f"  {title}", "bold", color))
        print(_RULE)
        for line in body.rstrip().splitlines():
            print(f"  {line}")
        print()


# ── 事件分发 ─────────────────────────────────────────────────────────────────

def on_event(ev: dict) -> None:
    """统一事件入口：按类型分发到各打印函数。"""
    t = ev.get("type")
    if t == "action":
        _print_action(ev)
    elif t == "observation":
        _print_observation(ev)
    elif t == "thought":
        _print_thought(ev)
    elif t == "thought_delta":
        _thought_delta(ev)
    elif t == "final_delta":
        _stream_delta(ev)
    elif t == "tool_call_seen":
        _on_tool_call_seen(ev)
    elif t == "final":
        if ev.get("streamed"):
            _stream_flush(ev.get("subagent_id") or ev["agent"])
        elif ev.get("agent") == "orchestrator":
            _print_synthesis(ev["answer"])
        else:
            _print_subagent_final(ev)
    elif t == "error":
        _print_error(ev)
    elif t == "dispatch_start":
        _print_dispatch_start(ev["tasks"])
    elif t == "subagent_start":
        _print_subagent_start(ev["task"])
    elif t == "subagent_done":
        _print_subagent_done(ev["result"])
    elif t == "dispatch_report":
        _print_dispatch_report(ev["results"])


# ── 主 Agent ReAct 过程（Thought 思维链 + 步骤横幅） ────────────────────────
# 思考来自模型的 reasoning_content 字段，在流式响应中先于正文到达，
# 因此能按 思考 → 行动 → 观察 → 最终答案 的顺序实时呈现完整 ReAct 轨迹。

_ORCH_STEP = 0                      # 主 agent 当前步骤，用于打步骤横幅
_THOUGHTS: dict[str, dict] = {}     # agent key -> {"step": int, "buf": str, "header": bool}


def _step_banner(step: int) -> None:
    """每个决策步骤打一条分隔横幅。"""
    global _ORCH_STEP
    if step != _ORCH_STEP:
        _ORCH_STEP = step
        print()
        print(_c(f"━━━ 主 Agent ReAct Step {step} ━━━", "bold", "cyan"))


def _print_thought(ev: dict) -> None:
    """非流式完整思考（subagent 不展示，只展示主 agent）。"""
    if ev.get("agent") != "orchestrator":
        return
    _step_banner(ev["step"])
    print(_c("  🧠 Thought:", "bold"))
    for line in ev["thought"].splitlines():
        print(f"    {line}")


def _thought_delta(ev: dict) -> None:
    """流式思考：边生成边打印完整行。"""
    if ev.get("agent") != "orchestrator":
        return
    key = ev["agent"]
    with _LOCK:
        st = _THOUGHTS.get(key)
        if st is None or st["step"] != ev["step"]:
            st = {"step": ev["step"], "buf": "", "header": False}
            _THOUGHTS[key] = st
        _step_banner(ev["step"])
        if not st["header"]:
            st["header"] = True
            print(_c("  🧠 Thought:", "bold"))
        st["buf"] += ev["delta"]
        while "\n" in st["buf"]:
            line, st["buf"] = st["buf"].split("\n", 1)
            print(f"    {line}")


def _flush_thought(key: str) -> None:
    """思考结束：打印剩余半行并清缓冲（进入 Action / 最终答案前调用，须持锁）。"""
    st = _THOUGHTS.pop(key, None)
    if st and st["buf"]:
        print(f"    {st['buf']}")


# ── 流式输出（最终答案边生成边打印） ────────────────────────────────────────
# subagent 并行时多个流交错，按"整行"加锁打印，行内不会串字。
# key = subagent_id（task-x）或 "orchestrator"

_STREAMS: dict[str, dict] = {}   # key -> {"buf": str, "sub": bool}


def _stream_delta(ev: dict) -> None:
    key = ev.get("subagent_id") or ev["agent"]
    with _LOCK:
        st = _STREAMS.get(key)
        if st is None:
            st = {"buf": "", "sub": bool(ev.get("subagent_id")), "header": False}
            _STREAMS[key] = st
            if st["sub"]:
                print(_c(f"     ── [{key}] 最终结果（流式输出中）", "cyan"))
        st["buf"] += ev["delta"]
        # 未打标题时先积累：出现完整行才提交为"最终答案"并打标题，
        # 避免把"调用工具前的决策前言"误标成最终汇总（标题延迟约一个完整行）
        if not st["header"] and not st["sub"]:
            if "\n" not in st["buf"]:
                return
            _flush_thought(key)   # 思考在正文前到达，正文开始时思考已结束
            st["header"] = True
            print()
            print(_RULE)
            print(_c("  ✅ 最终汇总（orchestrator）· 流式输出", "bold", "magenta"))
            print(_RULE)
        _flush_completed_lines(key, st)


def _flush_completed_lines(key: str, st: dict) -> None:
    """把缓冲区里已完整的行（以 \n 结尾）打印出来，剩余半行留在缓冲。"""
    prefix = f"[{key}] " if st["sub"] else ""
    while "\n" in st["buf"]:
        line, st["buf"] = st["buf"].split("\n", 1)
        print(f"  {prefix}{line}")


def _stream_flush(key: str) -> None:
    """流结束（final 事件）：补标题、打印剩余半行并清理缓冲。"""
    with _LOCK:
        st = _STREAMS.pop(key, None)
        if st is None:
            return
        if not st["sub"] and not st["header"]:
            # 整段答案始终没有换行（短答案）：此刻才确认是最终答案，补标题
            _flush_thought(key)
            print()
            print(_RULE)
            print(_c("  ✅ 最终汇总（orchestrator）· 流式输出", "bold", "magenta"))
            print(_RULE)
        if st["buf"]:
            prefix = f"[{key}] " if st["sub"] else ""
            print(f"  {prefix}{st['buf']}")
        if not st["sub"]:
            print(_RULE)


def _on_tool_call_seen(ev: dict) -> None:
    """流中发现 tool_call：刚流过的文本是"决策前言"而非最终答案。"""
    if ev.get("agent") != "orchestrator":
        return
    key = ev["agent"]
    with _LOCK:
        st = _STREAMS.pop(key, None)
        if st is None:
            return
        _flush_thought(key)
        if not st["header"] and st["buf"].strip():
            # 前言尚未打标题：作为 📝 决策说明 收进 ReAct 轨迹
            print(_c(f"  📝 决策说明: {st['buf'].strip()}", "dim"))
        elif st["header"]:
            # 罕见情况：前言已含完整行、标题已打，标注纠正
            print(_c("  （注意：上方文本是模型调用工具前的说明，非最终答案）", "red"))


# ── 编排者主循环 ─────────────────────────────────────────────────────────────

def print_banner(question: str, model: str, max_workers: int) -> None:
    with _LOCK:
        print()
        print(_RULE)
        print(_c("  Agent Orchestrator · 可并行下发 subagent 的编排 Agent", "bold", "cyan"))
        print(_c(f"  模型: {model}    并行度: {max_workers}", "dim"))
        print(_c(f"  问题: {question}", "dim"))
        print(_RULE)


def _print_action(ev: dict) -> None:
    """工具调用轨迹的 Action 部分（执行前打印，长耗时工具执行期间可见）。"""
    sub = ev.get("subagent_id")
    tag = f"[{sub}] " if sub else ""
    with _LOCK:
        if sub:
            # subagent 轨迹（不流式、不展示思考）
            print()
            print(_c(f"  {tag}🧠 Step {ev['step']} 决策 → 调用工具 {ev['action']}", "bold"))
            print(_c(f"  {tag}🔧 Action:  {ev['action']}", "yellow"))
            print(_c(f"  {tag}   Input:   {ev['action_input']}", "yellow"))
            return

        # 主 agent ReAct 轨迹：先刷思考尾巴，再打步骤横幅
        _flush_thought(ev["agent"])
        # 防御：若流缓冲仍有残留（前言已打标题但 tool_call_seen 未触发），清掉并标注
        stale = _STREAMS.pop(ev["agent"], None)
        if stale is not None:
            print(_c("  （注意：上方文本是模型调用工具前的说明，非最终答案）", "red"))
        _step_banner(ev["step"])
        print(_c(f"  🔧 Action:  {ev['action']}", "yellow"))
        print(_c(
            f"     Input:   {json.dumps(ev['action_input'], ensure_ascii=False)[:400]}",
            "yellow",
        ))


def _print_observation(ev: dict) -> None:
    """工具调用轨迹的 Obs 部分（工具执行完后打印）。"""
    sub = ev.get("subagent_id")
    tag = f"[{sub}] " if sub else ""
    with _LOCK:
        if sub:
            print(_c(f"  {tag}👁  Obs:     {_trunc(ev['observation'], 800)}", "green"))
            return
        if ev["action"] == "dispatch_subagents":
            # 完整结果已在 subagent 报告块展示，这里给一句友好摘要；
            # 但工具实际返回错误时要如实展示
            if "错误" in ev["observation"][:30]:
                print(_c(f"  👁 Obs:     {_trunc(ev['observation'], 300)}", "red"))
            else:
                print(_c("  👁 Obs:     已收集全部子任务结果（详情见下方报告块）", "green"))
        else:
            print(_c(f"  👁 Obs:     {_trunc(ev['observation'], 500)}", "green"))


def _print_error(ev: dict) -> None:
    with _LOCK:
        print(_c(f"  ⚠️  [{ev.get('agent')}] {ev['answer']}", "red"))


def _print_synthesis(answer: str) -> None:
    _block("✅ 最终汇总（orchestrator）", answer, color="magenta")


# ── subagent 实时状态（并行交错） ────────────────────────────────────────────

def _print_dispatch_start(tasks: list[dict]) -> None:
    with _LOCK:
        print()
        print(_c(f"  📦 编排者下发 {len(tasks)} 个 subagent，并行执行中 …", "bold", "blue"))
        for t in tasks:
            print(_c(f"     · {t['id']} {t['name']}: {_trunc(t['goal'], 60)}", "blue"))


def _print_subagent_start(task: dict) -> None:
    with _LOCK:
        print(_c(f"     ⏳ subagent [{task['id']}] {task['name']} 启动", "blue"))


def _print_subagent_done(result: dict) -> None:
    with _LOCK:
        color = "green" if result["status"] == "ok" else "red"
        print(_c(
            f"     {'✓' if result['status'] == 'ok' else '✗'} subagent [{result['id']}] "
            f"{result['name']} 完成 · {result['duration_s']}s · {result['steps']} 步",
            color,
        ))


def _print_subagent_final(ev: dict) -> None:
    """subagent 的最终答案实时一行预览（完整内容在下方报告中）。"""
    with _LOCK:
        first = ev["answer"].strip().splitlines()[0] if ev["answer"].strip() else ""
        print(_c(f"     └─ [{ev.get('subagent_id')}] 结果首行: {_trunc(first, 80)}", "dim"))


# ── subagent 完整报告 ────────────────────────────────────────────────────────

def _print_dispatch_report(results: list[dict]) -> None:
    for r in results:
        status = r.get("status", "?")
        stat = (f"状态: {status} · 耗时 {r.get('duration_s')}s · {r.get('steps')} 步")
        lines = [f"目标: {_trunc(r.get('goal', ''), 200)}", stat]

        tool_calls = [s for s in r.get("trace", []) if s.get("kind") == "action"]
        if tool_calls:
            lines.append("")
            lines.append("工具调用:")
            for s in tool_calls:
                lines.append(
                    f"  [{s['step']}] {s['action']}({s['action_input']})"
                    f"\n      → {_trunc(s.get('observation', ''), 200)}"
                )

        final = (r.get("final") or "").strip()
        if final:
            # subagent 不流式，完整结果在报告块里一次性展示
            lines.append("")
            lines.append("最终结果（未完成）:" if r.get("failed") else "最终结果:")
            lines.extend(f"  {ln}" for ln in final.splitlines())

        title = f"subagent {r.get('id')} · {r.get('name')}"
        color = "green" if status == "ok" else "red"
        _block(title, "\n".join(lines), color=color)


# ── 收尾 ─────────────────────────────────────────────────────────────────────

def print_usage(usage) -> None:
    with _LOCK:
        print()
        print(_c(f"  📊 LLM 用量: {usage.summary()}", "dim"))
        print()
