"""
session.py — 单次会话的数据结构

职责：
  - 维护 messages 列表（OpenAI 兼容格式）
  - 生成 session_id、记录元数据（model / provider / 时间）
  - 提供 append_* / clear / to_dict / from_dict 等方法
"""

import uuid
from datetime import datetime
from typing import Any


def _new_session_id() -> str:
    return "s_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


class Session:
    def __init__(
        self,
        system_prompt: str,
        model: str,
        provider: str,
        session_id: str | None = None,
        messages: list[dict] | None = None,
        active_skills: list[str] | None = None,
    ):
        self.id = session_id or _new_session_id()
        self.model = model
        self.provider = provider
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.updated_at = self.created_at
        self._system_prompt = system_prompt
        self.messages: list[dict] = messages if messages is not None else [
            {"role": "system", "content": system_prompt}
        ]
        # 已激活的 Skill 列表（week13 新增）：按需加载，已加载的不再二次确认
        self.active_skills: list[str] = active_skills if active_skills is not None else []

    # ── 写入 ─────────────────────────────────────────────────────────────
    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._touch()

    def append_assistant_text(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._touch()

    def append_assistant_tool_calls(self, tool_calls: list[dict]) -> None:
        """tool_calls: [{'id':..., 'type':'function', 'function':{'name':..., 'arguments':...}}, ...]"""
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        })
        self._touch()

    def append_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._touch()

    def clear(self) -> None:
        """清空消息但保留 system prompt"""
        self.messages = [{"role": "system", "content": self._system_prompt}]
        self._touch()

    # ── Skill 激活追踪（week13）─────────────────────────────────────────
    def mark_skill_active(self, name: str) -> None:
        """标记 Skill 已激活（去重）"""
        if name not in self.active_skills:
            self.active_skills.append(name)

    def active_skill_names(self) -> list[str]:
        return list(self.active_skills)

    # ── 序列化 ───────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "id":            self.id,
            "created_at":    self.created_at,
            "updated_at":    self.updated_at,
            "model":         self.model,
            "provider":      self.provider,
            "messages":      self.messages,
            "active_skills": self.active_skills,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], default_system_prompt: str) -> "Session":
        msgs = list(d.get("messages") or [])
        # 缺 system prompt 时补一个
        if not msgs or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": default_system_prompt}] + msgs
        return cls(
            system_prompt=default_system_prompt,
            model=d.get("model", "unknown"),
            provider=d.get("provider", "unknown"),
            session_id=d.get("id"),
            messages=msgs,
            active_skills=d.get("active_skills") or [],
        )

    def _touch(self) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")
