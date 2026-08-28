"""agent.py - agent loop（复用 week12 run_turn）+ Skill 动态加载 + 工具动态注入。

流程（FR-008..014）：加载索引 -> LLM 选单个 Skill -> 加载主体 -> LLM 提取 ToolSpec
-> 动态注入 tools -> 经受限 Bash 执行。临时扩展绑定 Skill，轮次结束回收（FR-014）。
"""
from __future__ import annotations

import json
import os
import tempfile

from .bash_tool import BashExecutor
from .llm_client import PROVIDERS
from .logging import log_event, log_error
from .skill_loader import SkillLoader, ToolSpec, programs_in_template
from .whitelist import Whitelist, WhitelistExtension

MAX_STEPS = 10

SYSTEM_PROMPT = (
    "你是一名助手，可通过 bash 工具执行白名单内的 shell 命令（ls/cat/head/wc/grep 等）。"
    "非白名单或危险命令（删除/重定向/命令替换/解释器透传）会被拦截并返回结构化 BlockResult，"
        "此时请依据 reason 调整方案后重试。依据命令输出作答，不要编造。"
)

BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "执行白名单内的 shell 命令。非白名单/危险命令将被拦截并返回 BlockResult。",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "要执行的命令字符串"}},
            "required": ["command"],
        },
    },
}


def new_session() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


class Agent:
    def __init__(self, client, model: str, skills_dir: str = "skills",
                 whitelist: Whitelist | None = None):
        self.client = client
        self.model = model
        self.whitelist = whitelist or Whitelist.default()
        self.extension: WhitelistExtension | None = None
        self.executor = BashExecutor(self.whitelist)
        self.loader = SkillLoader(skills_dir)
        self.active_skill: str | None = None
        self.dynamic_tools: dict[str, ToolSpec] = {}

    # ── 工具列表 ──────────────────────────────────────────────
    def _tools_schema(self) -> list[dict]:
        schemas = [BASH_TOOL_SCHEMA]
        for spec in self.dynamic_tools.values():
            schemas.append(spec.to_function_schema())
        return schemas

    # ── Skill 选择 / 激活 / 停用 ─────────────────────────────
    def _maybe_pick_skill(self, question: str):
        entries = self.loader.load_index()
        if not entries:
            return None
        index_str = "\n".join(f"- {e.name}: {e.description}" for e in entries)
        prompt = (
            "可用技能：\n" + index_str +
            f"\n\n用户提问：{question}\n\n判断是否需要某个技能，至多命中一个。"
            '返回 JSON {"skill":"技能name"} 或 {"skill":null}。'
        )
        try:
            log_event("llm_request", purpose="skill_pick", question=question[:100])
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            name = data.get("skill")
            log_event("llm_response", purpose="skill_pick", skill=name, raw=content[:100])
        except (ValueError, AttributeError, TypeError) as e:
            log_error("skill_pick_failed", error=repr(e), question=question[:100])
            name = None
        if not name:
            return None
        for e in entries:
            if e.name == name:
                return e
        return None

    def _activate_skill(self, entry) -> None:
        body = self.loader.load_body(entry.path)
        specs = self.loader.extract_tools(self.client, self.model, body, entry.name)
        self.dynamic_tools = {s.name: s for s in specs}
        self.active_skill = entry.name
        # 临时扩展：捕获每个 command_template 中的所有程序（不止首个），FR-014
        ext_cmds: set[str] = set()
        for s in specs:
            ext_cmds |= programs_in_template(s.command_template)
        ext_cmds = {p for p in ext_cmds if p and p not in self.whitelist.commands}
        self.extension = WhitelistExtension(commands=ext_cmds, bound_skill=entry.name)
        self.executor.extension = self.extension
        log_event("skill_activated", skill=entry.name, tools=list(self.dynamic_tools),
                  extension=list(ext_cmds))

    def _deactivate_skill(self) -> None:
        if self.active_skill:
            log_event("skill_deactivated", skill=self.active_skill)
        self.dynamic_tools = {}
        self.active_skill = None
        self.extension = None
        self.executor.extension = None

    # ── 工具派发 ─────────────────────────────────────────────
    def _dispatch(self, name: str, args_json: str) -> str:
        args = json.loads(args_json or "{}")
        if name == "bash":
            return self.executor.run(args.get("command", ""))
        spec = self.dynamic_tools.get(name)
        if spec is None:
            return f"未知工具：{name}"
        tmp: str | None = None
        try:
            if spec.json_input:
                # 把结构化参数写为临时 JSON 文件，command_template 用 {__json__} 占位
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                                 encoding="utf-8") as tf:
                    json.dump(args, tf, ensure_ascii=False)
                    tmp = tf.name
                cmd = spec.command_template.format(**args, __json__=tmp)
            else:
                cmd = spec.command_template.format(**args)
            return self.executor.run(cmd)
        except (KeyError, IndexError, ValueError) as e:
            log_error("tool_render_failed", name=name, error=repr(e), args=args_json[:200])
            return f"工具参数渲染失败：{e}"
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError as e:
                    log_error("temp_cleanup_failed", path=tmp, error=repr(e))

    # ── agent loop（复用 week12）──────────────────────────────
    def run_turn(self, messages: list, question: str) -> str:
        messages.append({"role": "user", "content": question})
        picked = self._maybe_pick_skill(question)
        if picked is not None:
            self._activate_skill(picked)
        try:
            for step in range(1, MAX_STEPS + 1):
                log_event("llm_request", purpose="turn", step=step,
                          tools=len(self._tools_schema()))
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._tools_schema(),
                    tool_choice="auto",
                )
                msg = resp.choices[0].message
                log_event("llm_response", purpose="turn", step=step,
                          has_tool_calls=bool(msg.tool_calls),
                          content_len=len(msg.content or ""))
                if not msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    log_event("turn_done", steps=step, answer=(msg.content or "")[:200])
                    return msg.content or ""
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = tc.function.arguments
                    log_event("tool_call", name=name, args=args)
                    result = self._dispatch(name, args)
                    log_event("tool_result", name=name, output=result[:200])
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            messages.append({"role": "assistant", "content": "（达到最大步数，未给出最终回答）"})
            return "（达到最大步数，未给出最终回答）"
        finally:
            self._deactivate_skill()  # 任务结束回收扩展（FR-014）
