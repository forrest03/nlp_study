"""
Harness — 接入 DeepSeek 的对话型 agent（function calling 驱动）
================================================================

定位：一个能正常对话的 agent。skills 作为「工具」暴露给 LLM，由 LLM 在
完整对话上下文中自然决定是否调用某个 skill —— 不再有独立的 matcher 路由层。

记忆系统分层（见 memory.py）：
  - system_prompt.md        静态系统提示词
  - user_profile.json       跨 session 用户记忆
  - session_summaries.jsonl 历次 session 压缩摘要
  - current_session.jsonl   当前 session 完整对话（JSONL 追加写）
  - skill_state.json        skill 级状态 + interactions

工作流（agent loop）：
  1. memory.build_context_messages() 组装 system + 历史 + 当前输入
  2. 把所有 skills 的 frontmatter 契约转成 OpenAI function schema，作为 tools 传入
  3. 调 DeepSeek（带 tools）：
     - LLM 直接返回文本 → 这就是给用户的回复
     - LLM 返回 tool_calls → harness 执行对应 skill → 把结果作为 tool message
       回传 → 再次调 LLM 生成最终回复（可多轮 tool 调用）
  4. 新增消息追加到 current_session.jsonl
  5. 超阈值时自动压缩早期对话为摘要
  6. /quit 时归档 session 摘要 + 提取 user_profile

强依赖 DeepSeek（兼容 OpenAI function calling 格式）。
未配置 DEEPSEEK_API_KEY 时，CLI 在启动阶段即报错退出。

用法:
    python -m harness.cli                       # 对话 REPL
    python -m harness.cli "做 meticulous 闪卡"  # 单次请求（自然语言输出）
    python -m harness.cli --list                 # 列出 skills
    python -m harness.cli --history              # 查看记忆
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .loader import ProgressiveSkillLoader, SkillMeta, LoadedSkill
from .executor import GenericExecutor, ExecutionResult
from .memory import MemorySystem
from .config import HarnessConfig
from .llm import DeepSeekClient


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SKILLS_DIR = _PROJECT_ROOT / "harness" / "skills"
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"

# agent loop 中最多允许的工具调用轮数，防止异常情况下无限循环
_MAX_TOOL_ROUNDS = 5


class Harness:
    """对话型 agent：loader + executor + memory + llm，function calling 驱动。"""

    def __init__(
        self,
        skills_dir: Path = _DEFAULT_SKILLS_DIR,
        data_dir: Path = _DEFAULT_DATA_DIR,
        verbose: bool = True,
        config: Optional[HarnessConfig] = None,
    ):
        self.config = config or HarnessConfig.load()
        if not self.config.llm_available:
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未配置 —— harness 强依赖 LLM。\n"
                "请通过环境变量或 .env 文件配置（参考 .env.example）。"
            )

        self.loader = ProgressiveSkillLoader(skills_dir)
        self.memory = MemorySystem(data_dir)
        self.llm = DeepSeekClient(self.config)
        self.executor = GenericExecutor(
            self.memory, llm=self.llm, on_progress=self._on_skill_progress,
        )
        self.verbose = verbose
        self._registry_ready = False

    # ---- phase 1: registry --------------------------------------------
    def _ensure_registry(self) -> list[SkillMeta]:
        if self.verbose and not self._registry_ready:
            print(f"[loader] Phase 1: scanning {self.loader.skills_dir} ...")
        metas = self.loader.scan()
        if self.verbose and not self._registry_ready:
            print(f"[loader]   → {len(metas)} skill(s) registered (metadata only)")
        self._registry_ready = True
        return metas

    # ---- 对话型 agent loop (function calling) -------------------------
    def respond(self, user_input: str, *, stream: bool = True) -> str:
        """对话型 agent loop：LLM 在完整对话上下文中决定是否调用 skill。

        - memory.build_context_messages() 组装 system + 历史 + 当前输入
        - skills 作为 tools 传给 DeepSeek，LLM 自行判断是否调用
        - 调用 skill → 执行 → 结果作为 tool message 回传 → LLM 生成最终回复
        - 本轮新增消息追加到 current_session.jsonl（保留 tool_calls/tool 结果）
        - 超阈值时自动压缩早期对话为摘要

        stream=True 时，最终回复的文本 token 会实时打印到 stdout。
        """
        self._ensure_registry()

        # 用 memory 系统组装完整 context（system + 历史摘要 + token预算内对话 + 当前输入）
        messages = self.memory.build_context_messages(user_input)
        # 记录本轮新增消息的起始位置：
        # messages = [system, ...history, user_input]，user_input 是最后一条
        # 新增消息 = 当前 user 输入 + agent loop 产生的 assistant/tool 消息
        # 所以起点要 -1，把 user_input 也纳入记录范围
        new_messages_start = len(messages) - 1

        tools = self._build_tools()

        try:
            reply = self._run_agent_loop(messages, tools, user_input, stream=stream)
        except Exception as e:
            reply = f"抱歉，处理时出错了：{e}"
            messages.append({"role": "assistant", "content": reply})
            if stream:
                print(reply)

        # 只把本轮新增的消息追加到 session（user 输入 + assistant/tool 交互）
        # 这样 current_session.jsonl 持续增长，不重写旧内容
        self.memory.record_messages(messages[new_messages_start:])

        # 超阈值时压缩早期对话为摘要（控制 context 长度）
        if self.memory.compress_if_needed(self.llm):
            if self.verbose:
                print("  🗜️ 已自动压缩早期对话")
        return reply

    def _run_agent_loop(
        self, messages: list[dict], tools: list[dict], user_input: str,
        *, stream: bool = True,
    ) -> str:
        """核心循环：调 LLM → 有 tool_calls 就执行并回传 → 否则返回文本。

        stream=True 时，LLM 生成的文本回复实时打印到 stdout（工具调用阶段
        通常无文本输出，最终回复阶段才流式打印）。
        """
        on_delta = self._print_delta if stream else None

        for round_idx in range(_MAX_TOOL_ROUNDS):
            resp = self.llm.chat_with_tools(
                messages, tools, temperature=0.7,
                stream=stream, on_text_delta=on_delta,
            )
            # 把 assistant message（可能含 tool_calls）加回 messages
            messages.append(resp.message)

            if not resp.tool_calls:
                # LLM 直接给出文本回复（流式时已实时打印，这里补一个换行收尾）
                if stream:
                    print()
                return resp.text.strip()

            # 即将执行 skill 工具 —— 打印调用提示，让用户知道在干什么。
            # LLM 可能先输出了一段开场白（已流式打印），这里先换行再显示 🔧。
            if stream and resp.text.strip():
                print()
            for tc in resp.tool_calls:
                fn = tc.get("function") or {}
                skill_name = fn.get("name", "?")
                raw_args = fn.get("arguments", "{}")
                try:
                    args_preview = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError:
                    args_preview = {}
                args_str = ", ".join(f"{k}={v!r}" for k, v in args_preview.items()) or "(无参数)"
                print(f"🔧 调用 skill: {skill_name}  ({args_str})", flush=True)

            # 执行每个 tool_call，结果作为 tool message 回传
            for tc in resp.tool_calls:
                tool_reply = self._execute_tool_call(tc, user_input)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_reply,
                })
                # skill 执行完毕，最终回复阶段开始 —— 留个空行分隔进度与回复
                print(flush=True)

        msg = "(已达最大工具调用轮数，请稍后重试)"
        if stream:
            print(msg)
        return msg

    @staticmethod
    def _print_delta(text_delta: str) -> None:
        """流式回调：实时打印文本增量，不换行、立即刷新。"""
        print(text_delta, end="", flush=True)

    @staticmethod
    def _on_skill_progress(msg: str) -> None:
        """executor 进度回调：实时打印 skill 执行各阶段进度。"""
        print(msg, flush=True)

    def _execute_tool_call(self, tool_call: dict, user_input: str) -> str:
        """执行单个 tool_call，返回给 LLM 的结果文本。"""
        fn = tool_call.get("function") or {}
        skill_name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}

        meta = self.loader.get(skill_name)
        if not meta:
            return f"错误：未找到 skill '{skill_name}'"

        try:
            loaded: LoadedSkill = self.loader.load_full(meta)
            result: ExecutionResult = self.executor.execute(
                loaded, user_input, args=args,
            )
            return self._format_exec_result_for_llm(result)
        except Exception as e:
            return f"skill '{skill_name}' 执行异常：{e}"

    # ---- tools / messages 构建 ----------------------------------------
    def _build_tools(self) -> list[dict]:
        """把每个 skill 的 frontmatter 契约转成 OpenAI function schema。"""
        tools = []
        for m in self.loader.scan():
            tools.append({
                "type": "function",
                "function": {
                    "name": m.name,
                    "description": m.description,
                    "parameters": self._params_to_schema(m.params),
                },
            })
        return tools

    @staticmethod
    def _params_to_schema(params: list[dict]) -> dict:
        """把 SKILL.md 的 params 声明转成 JSON Schema properties。"""
        properties: dict[str, dict] = {}
        required: list[str] = []
        for p in params:
            name = p.get("name", "")
            if not name:
                continue
            properties[name] = {
                "type": p.get("type", "string"),
                "description": p.get("description", ""),
            }
            if p.get("required"):
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @staticmethod
    def _format_exec_result_for_llm(r: ExecutionResult) -> str:
        """把 ExecutionResult 压缩成 LLM 易读的摘要文本，作为 tool message 内容。"""
        parts = [f"phase={r.phase}, ok={r.ok}"]
        if r.invocation:
            if r.invocation.artifacts:
                parts.append(f"artifacts: {r.invocation.artifacts}")
            if r.invocation.description:
                parts.append(f"desc: {r.invocation.description}")
        if r.stdout:
            parts.append(f"stdout:\n{r.stdout[-800:]}")
        if r.stderr:
            parts.append(f"stderr:\n{r.stderr[-800:]}")
        if r.exit_code is not None:
            parts.append(f"exit_code: {r.exit_code}")
        if r.note:
            parts.append(f"note:\n{r.note}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_HELP = """\
Harness 命令:
  /list            列出所有已注册 skills (Phase 1)
  /history         查看执行历史 (最近 20 条)
  /state [name]    查看 skill 级记忆状态
  /memory          查看记忆系统状态（user_profile / session 摘要 / token）
  /show <name>     打印某 skill 的完整 SKILL.md
  /reload          重新扫描 skills 目录
  /summary         打印记忆汇总
  /usage           打印 LLM 调用统计
  /help            显示此帮助
  /quit            退出（自动归档会话 + 提取用户记忆）
其余输入进入对话：你可以正常聊天，也可以请求执行 skill。
"""


def _print_memory_status(harness: Harness) -> None:
    """打印记忆系统各分层的状态。"""
    m = harness.memory
    # user_profile
    profile = m.user_profile.load()
    facts = profile.get("facts", [])
    prefs = profile.get("preferences", {})
    print("  ── 用户记忆 (user_profile.json) ──")
    if facts:
        for f in facts:
            print(f"    事实: {f}")
    else:
        print("    (空)")
    if prefs:
        for k, v in prefs.items():
            print(f"    偏好: {k} = {v}")

    # session summaries
    summaries = m.summaries.recent()
    print(f"  ── 历史会话摘要 ({len(summaries)} 条) ──")
    for s in summaries[-3:]:
        ts = s.get("ts", "")[:16]
        text = s.get("summary", "")[:80]
        print(f"    [{ts}] {text}...")

    # current session
    current_summary = m.session.get_summary()
    msgs = m.session.get_messages()
    total_tokens = m.session.total_tokens()
    print(f"  ── 当前会话 (current_session.jsonl) ──")
    print(f"    消息数: {len(msgs)}, 估算 token: {total_tokens}")
    if current_summary:
        print(f"    已有摘要: {current_summary[:80]}...")
    else:
        print("    已有摘要: (无)")


def run_repl(harness: Harness) -> None:
    print("=" * 72)
    print(f" Harness Agent — 对话模式  (DeepSeek-powered)")
    print(f" model: {harness.config.deepseek_model}  "
          f"endpoint: {harness.config.deepseek_base_url}")
    print("=" * 72)
    metas = harness._ensure_registry()
    if metas:
        print("可用 skills:")
        for m in metas:
            desc = m.description[:70] + "..." if len(m.description) > 70 else m.description
            print(f"  - {m.name}  v{m.version}  {desc}")
    print()
    print(_HELP)

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C 退出时也归档
            msgs = harness.memory.session.get_messages()
            if msgs:
                print(f"\n  归档会话（{len(msgs)} 条消息）...", flush=True)
                harness.memory.finalize_session(harness.llm)
                print("  ✓ 会话已归档")
            print("bye.")
            break
        if not line:
            continue

        if not line.startswith("/"):
            try:
                # respond 流式输出已实时打印，这里只补一个分隔换行
                harness.respond(line)
                print()
            except Exception as e:
                print(f"[error] {e!r}")
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            # session 结束：归档摘要 + 提取 user_profile + 清空当前 session
            msgs = harness.memory.session.get_messages()
            if msgs:
                print(f"  归档会话（{len(msgs)} 条消息）...", flush=True)
                harness.memory.finalize_session(harness.llm)
                print("  ✓ 会话已归档，用户记忆已更新")
            print("bye.")
            break
        elif cmd == "/help":
            print(_HELP)
        elif cmd == "/list":
            for m in harness.loader.scan():
                desc = m.description[:60] + "..." if len(m.description) > 60 else m.description
                print(f"  - {m.name}  v{m.version}  {desc}")
        elif cmd == "/history":
            for item in harness.memory.history(limit=20):
                ts = item["ts"][11:]
                skill = item.get("skill") or "-"
                phase = item.get("phase", "?")
                inp = (item.get("user_input") or "")[:40]
                print(f"  {ts} [{phase:11}] {skill:15} | {inp}")
        elif cmd == "/state":
            state = harness.memory.skills.all_state()
            if arg:
                print(f"  {arg}: {state.get(arg, {})}")
            else:
                for k, v in state.items():
                    print(f"  {k}: {v}")
        elif cmd == "/memory":
            _print_memory_status(harness)
        elif cmd == "/show":
            if not arg:
                print("  usage: /show <skill-name>")
                continue
            meta = harness.loader.get(arg)
            if not meta:
                print(f"  not found: {arg}")
                continue
            loaded = harness.loader.load_full(meta)
            print(loaded.body)
        elif cmd == "/reload":
            n = len(harness.loader.scan(force=True))
            harness._registry_ready = True
            print(f"  reloaded: {n} skill(s)")
        elif cmd == "/summary":
            s = harness.memory.summary()
            print(f"  total_interactions: {s['total_interactions']}")
            print(f"  skills_with_state: {s['skills_with_state']}")
            print(f"  per_skill_runs: {s['per_skill_runs']}")
        elif cmd == "/usage":
            u = harness.llm.usage
            print(f"  calls:              {u.calls}")
            print(f"  prompt_tokens:      {u.prompt_tokens}")
            print(f"  completion_tokens:   {u.completion_tokens}")
            print(f"  total_tokens:        {u.total_tokens}")
            print(f"  last_latency_ms:    {u.last_latency_ms}")
        else:
            print(f"  unknown command: {cmd} (try /help)")


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Progressive skill loading & execution harness (DeepSeek-powered)",
    )
    parser.add_argument("--skills-dir", type=Path, default=_DEFAULT_SKILLS_DIR,
                        help=f"skills 目录 (default: {_DEFAULT_SKILLS_DIR})")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
                        help=f"记忆数据目录 (default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--model", default=None,
                        help="覆盖 DEEPSEEK_MODEL (默认: deepseek-v4-flash)")
    parser.add_argument("--quiet", action="store_true",
                        help="减少 loader/matcher 日志")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="仅列出 skills 后退出")
    group.add_argument("--history", action="store_true", help="打印历史后退出")
    group.add_argument("--summary", action="store_true", help="打印记忆汇总后退出")
    group.add_argument("--usage", action="store_true", help="打印 LLM 调用统计后退出")
    group.add_argument("request", nargs="*", help="单次请求文本（不进入 REPL）")

    args = parser.parse_args(argv)

    config = HarnessConfig.load()
    if args.model:
        config.deepseek_model = args.model

    # 早期校验：未配置 key 直接报错退出
    if not config.llm_available:
        print(
            "ERROR: DEEPSEEK_API_KEY 未配置 —— harness 强依赖 LLM。\n"
            "请通过环境变量或 .env 文件配置（参考 .env.example）。",
            file=sys.stderr,
        )
        return 2

    try:
        harness = Harness(
            skills_dir=args.skills_dir,
            data_dir=args.data_dir,
            verbose=not args.quiet,
            config=config,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.list:
        metas = harness._ensure_registry()
        if args.quiet:
            for m in metas:
                print(f"{m.name}\t{m.version}\t{m.description[:60]}")
        else:
            print(f"{len(metas)} skill(s) registered:")
            for m in metas:
                print(f"  - {m.name}  v{m.version}")
                print(f"      {m.description[:120]}")
        return 0

    if args.history:
        items = harness.memory.history(limit=50)
        if not items:
            print("(no history yet)")
            return 0
        for item in items:
            ts = item["ts"]
            skill = item.get("skill") or "-"
            phase = item.get("phase", "?")
            inp = (item.get("user_input") or "")[:50]
            print(f"{ts} [{phase:11}] {skill:15} | {inp}")
        return 0

    if args.summary:
        print(harness.memory.summary())
        return 0

    if args.usage:
        u = harness.llm.usage
        print(f"calls: {u.calls}")
        print(f"prompt_tokens: {u.prompt_tokens}")
        print(f"completion_tokens: {u.completion_tokens}")
        print(f"total_tokens: {u.total_tokens}")
        print(f"last_latency_ms: {u.last_latency_ms}")
        return 0

    if args.request:
        req = " ".join(args.request)
        # respond 流式输出已实时打印，返回值仅用于退出码判断
        reply = harness.respond(req)
        return 0 if reply else 1

    run_repl(harness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
