"""
session_store.py — 会话 JSON 持久化

约定：
  - 每个 session 存为 <sessions_dir>/<id>.json
  - JSON 内含 id / created_at / updated_at / model / provider / messages
  - sessions_dir 不存在时自动创建
"""

import json
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, sessions_dir: Path | str):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def save(self, session_id: str, payload: dict[str, Any]) -> None:
        """保存一个 session；目录不存在会自动创建。"""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> dict[str, Any] | None:
        """加载一个 session；不存在返回 None。"""
        path = self._path(session_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self) -> list[str]:
        """列出所有 session id（按 id 字符串排序）。"""
        if not self.sessions_dir.exists():
            return []
        return sorted(p.stem for p in self.sessions_dir.glob("*.json"))
