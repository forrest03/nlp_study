"""
记忆存储模块

Markdown 记忆文件（memory/）：
  - short_term/session_{id}.md   短期记忆：当前会话最近 N 轮对话
  - long_term/memories_raw.md    长期记忆原始缓冲区（压缩前）
  - compressed/memories.md       长期记忆压缩结果（人类可读）
  - user_profile/profile.md      用户特征记忆
  - daily/YYYY-MM-DD.md          按日期记忆

检索数据库（databases/，由 memory_compressor 维护）：
  - memory_meta.json             BM25/RAG 检索元数据
  - memory_faiss_index.bin       FAISS 向量索引
"""

import json
import re
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.paths import (
    COMPRESSED_MD,
    COMPRESSED_MD_DIR,
    DAILY_DIR,
    DATABASE_DIR,
    LONG_TERM_DIR,
    LONG_TERM_RAW_MD,
    MEMORY_META_FILE,
    SHORT_TERM_DIR,
    USER_PROFILE_DIR,
    USER_PROFILE_MD,
)

SHORT_TERM_MAX_TURNS = 20

_TURN_HEADER_RE = re.compile(
    r"^### 轮次 \d+ · (\d{2}:\d{2}:\d{2})\s*$",
    re.MULTILINE,
)


def _ensure_dirs():
    for d in (
        SHORT_TERM_DIR,
        LONG_TERM_DIR,
        COMPRESSED_MD_DIR,
        USER_PROFILE_DIR,
        DAILY_DIR,
        DATABASE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_time(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


class MemoryStore:
    """统一读写各类 Markdown 记忆文件。"""

    def __init__(self):
        _ensure_dirs()

    # ── 短期记忆 ──────────────────────────────────────────────────────────────

    def _short_term_path(self, session_id: str) -> Path:
        return SHORT_TERM_DIR / f"session_{session_id}.md"

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        self._save_short_term(session_id, [])
        return session_id

    def _render_short_term_md(self, session_id: str, turns: List[Dict[str, Any]]) -> str:
        updated = _fmt_ts(turns[-1]["timestamp"]) if turns else _fmt_ts(int(time.time()))
        lines = [
            f"# 短期记忆 · 会话 `{session_id}`",
            "",
            f"> 更新时间：{updated}  ",
            f"> 说明：保留最近 {SHORT_TERM_MAX_TURNS} 轮对话，供当前会话上下文使用。",
            "",
            "## 对话记录",
            "",
        ]
        if not turns:
            lines.append("_（暂无对话）_")
        else:
            for i, turn in enumerate(turns, 1):
                lines.extend([
                    f"### 轮次 {i} · {_fmt_time(turn['timestamp'])}",
                    "",
                    f"**用户**：{turn['question']}",
                    "",
                    f"**助手**：{turn['answer']}",
                    "",
                ])
        return "\n".join(lines) + "\n"

    def _parse_short_term_md(self, text: str) -> List[Dict[str, Any]]:
        turns: List[Dict[str, Any]] = []
        blocks = re.split(r"(?=^### 轮次 \d+ · )", text, flags=re.MULTILINE)
        for block in blocks:
            if not block.strip().startswith("### 轮次"):
                continue
            header = _TURN_HEADER_RE.search(block)
            if not header:
                continue
            h, m, s = map(int, header.group(1).split(":"))
            today = date.today()
            ts = int(datetime(today.year, today.month, today.day, h, m, s).timestamp())

            q_match = re.search(r"\*\*用户\*\*：(.+?)(?=\n\n\*\*助手\*\*|\Z)", block, re.DOTALL)
            a_match = re.search(r"\*\*助手\*\*：(.+?)(?=\n\n### |\Z)", block, re.DOTALL)
            if q_match and a_match:
                turns.append({
                    "question": q_match.group(1).strip(),
                    "answer": a_match.group(1).strip(),
                    "timestamp": ts,
                })
        return turns

    def _save_short_term(self, session_id: str, turns: List[Dict[str, Any]]):
        path = self._short_term_path(session_id)
        trimmed = turns[-SHORT_TERM_MAX_TURNS:]
        path.write_text(
            self._render_short_term_md(session_id, trimmed),
            encoding="utf-8",
        )

    def get_short_term(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._short_term_path(session_id)
        if not path.exists():
            return []
        return self._parse_short_term_md(path.read_text(encoding="utf-8"))

    def add_short_term_turn(self, session_id: str, question: str, answer: str):
        turns = self.get_short_term(session_id)
        turns.append({
            "question": question,
            "answer": answer,
            "timestamp": int(time.time()),
        })
        self._save_short_term(session_id, turns)
        self.add_long_term_raw(f"[会话 {session_id}] 问：{question}\n答：{answer}")
        preview = answer[:200] + ("..." if len(answer) > 200 else "")
        self.append_daily(f"问：{question}\n答：{preview}")

    def short_term_to_messages(self, session_id: str) -> List[Dict[str, str]]:
        messages = []
        for turn in self.get_short_term(session_id):
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        return messages

    # ── 长期记忆（原始 Markdown） ─────────────────────────────────────────────

    def _ensure_long_term_raw_header(self):
        if not LONG_TERM_RAW_MD.exists():
            LONG_TERM_RAW_MD.write_text(
                "# 长期记忆原始缓冲区\n\n"
                "> 对话结束后逐条追加于此；达到阈值后由压缩模块分析并写入 `compressed/memories.md`，"
                "同时更新 `databases/` 检索索引。\n\n",
                encoding="utf-8",
            )

    def add_long_term_raw(self, content: str):
        self._ensure_long_term_raw_header()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        block = f"\n## {ts}\n\n{content.strip()}\n"
        with open(LONG_TERM_RAW_MD, "a", encoding="utf-8") as f:
            f.write(block)

    def read_long_term_raw(self) -> str:
        if not LONG_TERM_RAW_MD.exists():
            return ""
        text = LONG_TERM_RAW_MD.read_text(encoding="utf-8")
        # 去掉文件头，只返回条目内容用于压缩
        parts = re.split(r"\n## \d{4}-\d{2}-\d{2}", text)
        if len(parts) <= 1:
            return ""
        entries = re.findall(
            r"## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\n\n(.*?)(?=\n## |\Z)",
            text,
            re.DOTALL,
        )
        return "\n\n".join(f"--- [{ts}] ---\n{body.strip()}" for ts, body in entries)

    def read_long_term_raw_md(self) -> str:
        """读取长期记忆原始 Markdown 全文（供 Web 编辑）。"""
        self._ensure_long_term_raw_header()
        return LONG_TERM_RAW_MD.read_text(encoding="utf-8")

    def write_long_term_raw_md(self, content: str):
        """写入长期记忆原始 Markdown 全文。"""
        LONG_TERM_RAW_MD.write_text(content.strip() + "\n", encoding="utf-8")

    def clear_long_term_raw(self):
        self._ensure_long_term_raw_header()

    # ── 长期记忆（压缩 Markdown + 检索元数据） ────────────────────────────────

    def read_compressed_chunks(self) -> List[Dict[str, Any]]:
        """从 databases/memory_meta.json 读取检索块。"""
        if not MEMORY_META_FILE.exists():
            return []
        return json.loads(MEMORY_META_FILE.read_text(encoding="utf-8"))

    def save_compressed_chunks(self, chunks: List[Dict[str, Any]]):
        """追加检索块到 databases/，并同步写入 memory/compressed/memories.md。"""
        existing = self.read_compressed_chunks()
        existing.extend(chunks)
        MEMORY_META_FILE.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._append_compressed_md(chunks)

    def _ensure_compressed_md_header(self):
        if not COMPRESSED_MD.exists():
            COMPRESSED_MD.write_text(
                "# 压缩长期记忆\n\n"
                "> LLM 定期分析 `long_term/memories_raw.md` 后生成的摘要，供人工查阅。"
                "BM25/RAG 检索使用 `databases/memory_meta.json`。\n\n",
                encoding="utf-8",
            )

    def _append_compressed_md(self, chunks: List[Dict[str, Any]]):
        self._ensure_compressed_md_header()
        lines = []
        for chunk in chunks:
            tags = ", ".join(chunk.get("tags", []))
            created = _fmt_ts(chunk.get("created_at", int(time.time())))
            lines.extend([
                f"## {chunk['chunk_id']}",
                "",
                f"- **标签**：{tags or '无'}",
                f"- **创建时间**：{created}",
                f"- **内容**：{chunk['content']}",
                "",
            ])
        with open(COMPRESSED_MD, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def compressed_needs_update(self, min_raw_chars: int = 500) -> bool:
        raw = self.read_long_term_raw()
        return len(raw.strip()) >= min_raw_chars

    def read_compressed_md(self) -> str:
        """读取压缩长期记忆 Markdown 全文。"""
        self._ensure_compressed_md_header()
        return COMPRESSED_MD.read_text(encoding="utf-8")

    def write_compressed_md(self, content: str):
        COMPRESSED_MD.write_text(content.strip() + "\n", encoding="utf-8")

    def read_memory_meta_raw(self) -> str:
        if not MEMORY_META_FILE.exists():
            return "[]"
        return MEMORY_META_FILE.read_text(encoding="utf-8")

    def write_memory_meta_raw(self, content: str):
        chunks = json.loads(content)
        MEMORY_META_FILE.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有短期记忆会话。"""
        sessions = []
        for path in sorted(SHORT_TERM_DIR.glob("session_*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            sid = path.stem.replace("session_", "")
            if sid == "example":
                continue
            turns = self.get_short_term(sid)
            preview = turns[-1]["question"][:40] if turns else "（空会话）"
            sessions.append({
                "session_id": sid,
                "turn_count": len(turns),
                "preview": preview,
                "updated_at": path.stat().st_mtime,
            })
        return sessions

    # ── 用户特征记忆 ──────────────────────────────────────────────────────────

    def read_user_profile(self) -> str:
        if not USER_PROFILE_MD.exists():
            return ""
        return USER_PROFILE_MD.read_text(encoding="utf-8")

    def update_user_profile(self, content: str):
        USER_PROFILE_MD.write_text(content.strip() + "\n", encoding="utf-8")

    # ── 按日期记忆 ────────────────────────────────────────────────────────────

    def _daily_path(self, day: Optional[date] = None) -> Path:
        d = day or date.today()
        return DAILY_DIR / f"{d.isoformat()}.md"

    def _ensure_daily_header(self, path: Path, day: date):
        if not path.exists():
            path.write_text(
                f"# {day.isoformat()} 按日期记忆\n\n"
                f"> 记录当天对话摘要与事件。\n\n",
                encoding="utf-8",
            )

    def append_daily(self, content: str, day: Optional[date] = None):
        d = day or date.today()
        path = self._daily_path(d)
        self._ensure_daily_header(path, d)
        ts = time.strftime("%H:%M:%S")
        block = f"## {ts}\n\n{content.strip()}\n\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)

    def read_daily(self, day: Optional[date] = None) -> str:
        path = self._daily_path(day)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_daily(self, content: str, day: Optional[date] = None):
        d = day or date.today()
        path = self._daily_path(d)
        path.write_text(content.strip() + "\n", encoding="utf-8")

    def list_daily_files(self) -> List[Path]:
        return sorted(DAILY_DIR.glob("*.md"))


_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
