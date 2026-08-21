"""
memory — 分层记忆系统
========================

数据文件分层（均在 data/ 目录下）：
  system_prompt.md          静态系统提示词（角色定位 + 工具准则，基本不变）
  user_profile.json         跨 session 用户记忆（身份、偏好、长期事实）
  session_summaries.jsonl   历次 session 压缩摘要（每行一个 JSON，便于追加）
  current_session.jsonl     当前 session 完整对话（每行一条消息，JSONL 追加写）
  skill_state.json          skill 级状态 + interactions 历史

拼接给 LLM 的 context 结构：
  [system]
    system_prompt.md 内容
    + user_profile 摘要
    + 最近 N 个 session 摘要（跨 session 记忆）
    + 当前 session 摘要（动态压缩更新）
  [messages]
    最近 token 预算内的完整对话（含 tool_calls/tool 成对性保护）

token 控制：
  - 完整对话预算: 4000 token（超 5000 时触发压缩）
  - 估算方式: 中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ---- token 预算配置 -------------------------------------------------
_DIALOG_TOKEN_BUDGET = 4000   # 拼入 prompt 的完整对话 token 上限
_COMPRESS_THRESHOLD = 5000   # 超过此值触发压缩
_KEEP_RECENT_TOKENS = 1500   # 压缩时保留最近这么多 token 的完整对话
_MAX_SESSION_SUMMARIES = 5   # 拼入 prompt 的历史 session 摘要数


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    中文 1 字 ≈ 1 token，英文/符号 4 字符 ≈ 1 token。
    对中英混合内容比较保守（偏高），确保不超 context 上限。
    """
    if not text:
        return 0
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    non_chinese = len(text) - chinese_count
    return int(chinese_count + non_chinese * 0.25)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# =====================================================================
# 1. SystemPromptStore — 静态系统提示词
# =====================================================================
class SystemPromptStore:
    """加载 data/system_prompt.md，基本不变，支持热重载。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cache: Optional[str] = None

    def load(self) -> str:
        if self._cache is None and self.path.exists():
            self._cache = self.path.read_text(encoding="utf-8").strip()
        return self._cache or ""

    def reload(self) -> str:
        self._cache = None
        return self.load()


# =====================================================================
# 2. UserProfileStore — 跨 session 用户记忆
# =====================================================================
class UserProfileStore:
    """data/user_profile.json — 用户身份、偏好、长期事实。

    格式:
    {
      "facts": ["用户叫小王", "用户在学英语单词"],
      "preferences": {"style": "简洁", "language": "zh"},
      "updated": "2026-07-30T22:00:00"
    }
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict = {"facts": [], "preferences": {}}

    def load(self) -> dict:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {"facts": [], "preferences": {}}
        self._data.setdefault("facts", [])
        self._data.setdefault("preferences", {})
        return self._data

    def save(self) -> None:
        self._data["updated"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def merge(self, update: dict) -> None:
        """合并 LLM 提取的新信息到现有 profile。"""
        self.load()
        new_facts = update.get("facts") or []
        existing = set(self._data.get("facts", []))
        for f in new_facts:
            if f and f not in existing:
                self._data["facts"].append(f)
                existing.add(f)
        prefs = update.get("preferences") or {}
        self._data["preferences"].update(prefs)
        self.save()

    def to_prompt_section(self) -> str:
        """生成拼入 system prompt 的用户记忆段落。"""
        self.load()
        facts = self._data.get("facts", [])
        prefs = self._data.get("preferences", {})
        if not facts and not prefs:
            return ""
        parts = ["\n\n# 用户记忆（跨会话保留）"]
        if facts:
            parts.append("## 关于用户的事实:")
            for f in facts:
                parts.append(f"- {f}")
        if prefs:
            parts.append("## 用户偏好:")
            for k, v in prefs.items():
                parts.append(f"- {k}: {v}")
        return "\n".join(parts)


# =====================================================================
# 3. SessionStore — 当前 session 完整对话（JSONL 追加写）
# =====================================================================
class SessionStore:
    """data/current_session.jsonl — 当前 session 的完整对话消息。

    JSONL 格式（每行一个 JSON）：
      第一行（可选）: {"type": "summary", "content": "...", "ts": "..."}
      后续行: {"role": "user/assistant/tool", "content": "...",
              "tool_calls": [...], "tool_call_id": "...", "ts": "..."}

    追加写不读全文件；压缩时重写整个文件。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def _read_all_lines(self) -> list[dict]:
        """读取所有行，返回 dict 列表。"""
        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def _write_all(self, items: list[dict]) -> None:
        """重写整个文件（压缩时用）。"""
        lines = [json.dumps(item, ensure_ascii=False) for item in items]
        self.path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    def append_message(self, msg: dict) -> None:
        """追加一条对话消息到文件末尾。"""
        entry = dict(msg)
        entry["ts"] = _now()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def append_messages(self, msgs: list[dict]) -> None:
        """批量追加消息。"""
        ts = _now()
        with self.path.open("a", encoding="utf-8") as f:
            for m in msgs:
                entry = dict(m)
                entry["ts"] = ts
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_summary(self) -> str:
        """获取当前 session 的压缩摘要（第一行 type=summary）。"""
        items = self._read_all_lines()
        if items and items[0].get("type") == "summary":
            return items[0].get("content", "")
        return ""

    def set_summary(self, content: str) -> None:
        """更新当前 session 的摘要（重写第一行）。"""
        items = self._read_all_lines()
        summary_entry = {"type": "summary", "content": content, "ts": _now()}
        if items and items[0].get("type") == "summary":
            items[0] = summary_entry
        else:
            items.insert(0, summary_entry)
        self._write_all(items)

    def get_messages(self) -> list[dict]:
        """获取所有对话消息（去掉 summary 行）。"""
        items = self._read_all_lines()
        return [m for m in items if m.get("type") != "summary"]

    def get_all_with_summary(self) -> tuple[str, list[dict]]:
        """返回 (summary, messages)。"""
        return self.get_summary(), self.get_messages()

    def reset(self) -> None:
        """清空当前 session（finalize 后调用）。"""
        self._write_all([])

    def total_tokens(self) -> int:
        """估算所有对话消息的总 token 数。"""
        total = 0
        for m in self.get_messages():
            total += estimate_tokens(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                total += estimate_tokens(fn.get("arguments") or "")
        return total

    def recent_within_budget(self, max_tokens: int = _DIALOG_TOKEN_BUDGET) -> list[dict]:
        """返回 token 预算内的最近消息，保证 tool_calls/tool 成对性。

        从最近消息向前累加 token，超预算就停止。然后校验开头：
        - 跳过孤立的 tool 消息（前面无对应 assistant+tool_calls）
        - 跳过带 tool_calls 的 assistant 但后续无对应 tool 结果的情况
        """
        msgs = self.get_messages()
        if not msgs:
            return []

        # 从后向前累加 token
        selected: list[dict] = []
        used = 0
        for m in reversed(msgs):
            m_tokens = estimate_tokens(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                m_tokens += estimate_tokens(fn.get("arguments") or "")
            if used + m_tokens > max_tokens and selected:
                break
            selected.insert(0, m)
            used += m_tokens

        # 成对性保护：跳过开头孤立的消息
        selected = self._trim_unpaired_head(selected)
        return selected

    @staticmethod
    def _trim_unpaired_head(msgs: list[dict]) -> list[dict]:
        """跳过开头不成对的消息，确保 tool_calls 和 tool 结果成对。

        检查规则：
        1. 首条是 tool role → 跳过（前面无对应 assistant+tool_calls）
        2. 首条是 assistant 带 tool_calls，但后续无对应 tool 结果 → 跳过该 assistant
        """
        while msgs:
            first = msgs[0]
            role = first.get("role", "")

            # 规则 1：孤立的 tool 消息
            if role == "tool":
                msgs.pop(0)
                continue

            # 规则 2：assistant 带 tool_calls 但后续缺 tool 结果
            if role == "assistant" and first.get("tool_calls"):
                tool_call_ids = {tc.get("id") for tc in first["tool_calls"] if tc.get("id")}
                # 检查后续是否有对应的 tool 结果
                has_results = any(
                    m.get("role") == "tool" and m.get("tool_call_id") in tool_call_ids
                    for m in msgs[1:]
                )
                if not has_results:
                    msgs.pop(0)
                    continue

            break
        return msgs


# =====================================================================
# 4. SessionSummariesStore — 历次 session 压缩摘要
# =====================================================================
class SessionSummariesStore:
    """data/session_summaries.jsonl — 历次 session 的压缩摘要。

    每行一个 JSON: {"ts": "...", "summary": "...", "turns": N}
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, summary: str, turns: int = 0) -> None:
        entry = {"ts": _now(), "summary": summary, "turns": turns}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recent(self, limit: int = _MAX_SESSION_SUMMARIES) -> list[dict]:
        """返回最近 N 个 session 摘要。"""
        items = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items[-limit:] if items else []

    def to_prompt_section(self, limit: int = _MAX_SESSION_SUMMARIES) -> str:
        """生成拼入 system prompt 的历史摘要段落。"""
        items = self.recent(limit)
        if not items:
            return ""
        parts = ["\n\n# 历史会话摘要（跨会话记忆）"]
        for item in items:
            ts = item.get("ts", "")[:10]
            summary = item.get("summary", "")
            parts.append(f"## [{ts}]")
            parts.append(summary)
        return "\n".join(parts)


# =====================================================================
# 5. SkillStateStore — skill 级状态 + interactions 历史
# =====================================================================
class SkillStateStore:
    """data/skill_state.json — skill 执行记录与状态。

    格式:
    {
      "interactions": [...],
      "skill_state": {"flash-card": {"items_generated": [...], "last_item": {...}}}
    }
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = {"interactions": [], "skill_state": {}}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {"interactions": [], "skill_state": {}}
        self._data.setdefault("interactions", [])
        self._data.setdefault("skill_state", {})

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- interactions ---------------------------------------------------
    def record(
        self,
        user_input: str,
        skill: Optional[str],
        phase: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> dict:
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": _now(),
            "user_input": user_input,
            "skill": skill,
            "phase": phase,
            "result": result or {},
            "error": error,
        }
        self._data["interactions"].append(entry)
        self.save()
        return entry

    def history(self, limit: int = 20) -> list[dict]:
        items = self._data["interactions"]
        return list(reversed(items[-limit:]))

    # ---- per-skill state -----------------------------------------------
    def get_state(self, skill: str) -> dict:
        return self._data["skill_state"].setdefault(skill, {})

    def set_state(self, skill: str, state: dict) -> None:
        self._data["skill_state"][skill] = state
        self.save()

    def all_state(self) -> dict:
        return self._data["skill_state"]

    # ---- summary --------------------------------------------------------
    def summary(self) -> dict:
        return {
            "total_interactions": len(self._data["interactions"]),
            "skills_with_state": list(self._data["skill_state"].keys()),
            "per_skill_runs": {
                s: sum(1 for x in self._data["interactions"] if x.get("skill") == s)
                for s in {x.get("skill") for x in self._data["interactions"] if x.get("skill")}
            },
        }


# =====================================================================
# 6. MemorySystem — 统一入口
# =====================================================================
class MemorySystem:
    """组合所有记忆 store，提供统一的 context 组装与 session 生命周期管理。

    兼容旧 MemoryStore 接口（record/get_state/set_state/history/summary），
    executor.py 无需改动。
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.system_prompt = SystemPromptStore(self.data_dir / "system_prompt.md")
        self.user_profile = UserProfileStore(self.data_dir / "user_profile.json")
        self.session = SessionStore(self.data_dir / "current_session.jsonl")
        self.summaries = SessionSummariesStore(self.data_dir / "session_summaries.jsonl")
        self.skills = SkillStateStore(self.data_dir / "skill_state.json")

    # ---- 拼装 LLM context ----------------------------------------------
    def build_system_content(self) -> str:
        """组装完整的 system prompt：静态提示词 + 用户记忆 + 历史摘要 + 当前摘要。"""
        parts = [self.system_prompt.load()]
        parts.append(self.user_profile.to_prompt_section())
        parts.append(self.summaries.to_prompt_section())

        current_summary = self.session.get_summary()
        if current_summary:
            parts.append(f"\n\n# 当前会话摘要\n{current_summary}")

        return "\n".join(p for p in parts if p)

    def build_context_messages(self, user_input: str) -> list[dict]:
        """组装完整的 LLM messages：system + 历史 + 当前用户输入。

        历史消息按 token 预算截断，保证 tool_calls/tool 成对性。
        """
        messages: list[dict] = [
            {"role": "system", "content": self.build_system_content()}
        ]
        # token 预算内的最近对话
        history = self.session.recent_within_budget(_DIALOG_TOKEN_BUDGET)
        for m in history:
            # 去掉 ts 等非 LLM 字段
            msg = {k: v for k, v in m.items() if k != "ts"}
            # OpenAI 规范：tool_calls 消息 content 可为 None
            if msg.get("role") == "assistant" and msg.get("tool_calls") and not msg.get("content"):
                msg["content"] = None
            messages.append(msg)
        # 当前用户输入
        messages.append({"role": "user", "content": user_input})
        return messages

    # ---- session 消息记录（兼容旧接口）---------------------------------
    def record_messages(self, messages: list[dict]) -> None:
        """记录 agent loop 的完整消息序列到当前 session（追加写）。

        注意：这里只记录新增的消息（当前轮次产生的），不是全量覆盖。
        调用方应传入本轮新增的 messages（user + assistant + tool 等）。
        """
        self.session.append_messages(messages)

    def append_message(self, msg: dict) -> None:
        """追加单条消息到当前 session。"""
        self.session.append_message(msg)

    def get_turns(self, limit: int = 20) -> list[dict]:
        """兼容旧接口：返回最近 N 条消息。"""
        msgs = self.session.get_messages()
        return list(msgs[-limit:])

    # ---- session 压缩 --------------------------------------------------
    def compress_if_needed(self, llm) -> bool:
        """当当前 session 对话 token 超阈值时，压缩早期对话为摘要。

        llm: 需要有 .chat(messages, temperature, max_tokens) 接口的对象。
        返回是否执行了压缩。
        """
        total = self.session.total_tokens()
        if total < _COMPRESS_THRESHOLD:
            return False

        msgs = self.session.get_messages()
        if len(msgs) < 4:
            return False  # 消息太少不压缩

        # 从后向前保留 _KEEP_RECENT_TOKENS 的完整对话
        keep: list[dict] = []
        used = 0
        split_idx = len(msgs)
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            m_tokens = estimate_tokens(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                m_tokens += estimate_tokens(fn.get("arguments") or "")
            if used + m_tokens > _KEEP_RECENT_TOKENS and keep:
                split_idx = i + 1
                break
            keep.insert(0, m)
            used += m_tokens
        else:
            split_idx = 0

        to_compress = msgs[:split_idx]
        if not to_compress:
            return False

        # 成对性保护：确保 to_compress 末尾不是孤立的 assistant+tool_calls
        to_compress, orphaned = self._trim_unpaired_tail(to_compress)
        keep = orphaned + keep

        if not to_compress:
            return False

        # 调 LLM 压缩
        dialog_text = self._messages_to_text(to_compress)
        old_summary = self.session.get_summary()
        prompt = self._build_compress_prompt(dialog_text, old_summary)

        try:
            resp = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            new_summary = resp.text.strip()
        except Exception:
            return False  # 压缩失败不影响主流程

        # 更新 session：新 summary + 保留的消息
        self.session.set_summary(new_summary)
        # 重写消息部分（保留 keep）
        all_items = []
        if new_summary:
            all_items.append({"type": "summary", "content": new_summary, "ts": _now()})
        all_items.extend(keep)
        # 直接重写文件
        lines = [json.dumps(item, ensure_ascii=False) for item in all_items]
        self.session.path.write_text(
            "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
        )
        return True

    @staticmethod
    def _trim_unpaired_tail(msgs: list[dict]) -> tuple[list[dict], list[dict]]:
        """把末尾不成对的消息从 to_compress 移到 keep。

        如果 to_compress 末尾是 assistant+tool_calls 但后续无 tool 结果，
        把它移到 keep 列表。
        """
        keep: list[dict] = []
        while msgs:
            last = msgs[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                tool_call_ids = {tc.get("id") for tc in last["tool_calls"] if tc.get("id")}
                has_results = any(
                    m.get("role") == "tool" and m.get("tool_call_id") in tool_call_ids
                    for m in msgs[:-1]
                )
                if not has_results:
                    keep.insert(0, msgs.pop())
                    continue
            break
        return msgs, keep

    @staticmethod
    def _messages_to_text(msgs: list[dict]) -> str:
        """把消息序列转成 LLM 易读的对话文本。"""
        lines = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content") or ""
            if role == "tool":
                lines.append(f"[工具结果] {content[:500]}")
            elif role == "assistant" and m.get("tool_calls"):
                tc_names = [tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]]
                lines.append(f"[助手] 调用工具: {', '.join(tc_names)}")
                if content:
                    lines.append(f"  附言: {content[:200]}")
            elif role == "assistant":
                lines.append(f"[助手] {content[:800]}")
            else:
                lines.append(f"[{role}] {content[:500]}")
        return "\n".join(lines)

    @staticmethod
    def _build_compress_prompt(dialog_text: str, old_summary: str) -> str:
        prompt = "请把以下对话压缩成简洁的摘要（不超过 500 字），保留：\n"
        prompt += "- 用户的关键需求和意图\n- 执行了哪些 skill，结果如何\n"
        prompt += "- 重要的上下文信息（如用户身份、偏好、已生成的产物）\n"
        prompt += "去掉寒暄和冗余内容，用简洁的陈述句。\n\n"
        if old_summary:
            prompt += f"# 之前的摘要（请在此基础上增量更新）\n{old_summary}\n\n"
        prompt += f"# 待压缩的对话\n{dialog_text}\n\n请输出更新后的完整摘要："
        return prompt

    # ---- session 结束归档 ----------------------------------------------
    def finalize_session(self, llm) -> None:
        """session 结束时：压缩成摘要归档 + 提取 user_profile 更新 + 清空。

        llm: 需要有 .chat(messages, temperature, max_tokens) 接口。
        """
        summary, msgs = self.session.get_all_with_summary()
        if not msgs:
            return

        turns = len(msgs)
        dialog_text = self._messages_to_text(msgs)

        # 1. 生成最终 session 摘要并归档
        try:
            prompt = (
                "请把以下完整的会话压缩成简洁的摘要（不超过 500 字），保留：\n"
                "- 用户的关键需求和意图\n- 执行了哪些 skill，结果如何\n"
                "- 重要的上下文信息\n去掉寒暄和冗余内容。\n\n"
                f"# 当前会话已有摘要\n{summary}\n\n"
                f"# 完整对话\n{dialog_text}\n\n请输出最终摘要："
            )
            resp = llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            final_summary = resp.text.strip()
            self.summaries.append(final_summary, turns)
        except Exception:
            pass  # 归档失败不阻断

        # 2. 提取 user_profile 更新
        try:
            profile_prompt = (
                "请从以下对话中提取应该长期记住的用户信息，输出 JSON：\n"
                '{"facts": ["..."], "preferences": {"...": "..."}}\n'
                "只提取明确提到的信息（如用户姓名、身份、学习目标、偏好等），"
                "不要推测。如果没有值得长期记住的信息，输出空 JSON：{}\n\n"
                f"# 对话内容\n{dialog_text}"
            )
            resp = llm.chat_json(
                messages=[{"role": "user", "content": profile_prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            if resp.json_obj:
                self.user_profile.merge(resp.json_obj)
        except Exception:
            pass  # 提取失败不阻断

        # 3. 清空当前 session
        self.session.reset()

    # ---- 兼容旧 MemoryStore 接口（executor.py 用）---------------------
    def record(
        self,
        user_input: str,
        skill: Optional[str],
        phase: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> dict:
        return self.skills.record(user_input, skill, phase, result, error)

    def get_state(self, skill: str) -> dict:
        return self.skills.get_state(skill)

    def set_state(self, skill: str, state: dict) -> None:
        self.skills.set_state(skill, state)

    def history(self, limit: int = 20) -> list[dict]:
        return self.skills.history(limit)

    def summary(self) -> dict:
        return self.skills.summary()

    @property
    def _data(self) -> dict:
        """兼容旧代码直接访问 _data['skill_state']。"""
        return self.skills._data
