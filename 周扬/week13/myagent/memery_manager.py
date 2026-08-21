"""
记忆管理器。

这个模块负责三类长期记忆和一类会话记忆：
1. soul.md：智能体人格、行为准则
2. user.md：用户画像、长期偏好
3. memery.md：跨会话长期记忆
4. sessions / session_summaries：当前会话消息和摘要

这里不再负责向量库写入。
长期记忆更新统一通过专门的 update_* 函数来做。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).parent
SOUL_FILE = BASE_DIR / "memery" / "soul.md"
USER_FILE = BASE_DIR / "memery" / "user.md"
MEMERY_FILE = BASE_DIR / "memery" / "memery.md"
SESSION_DIR = BASE_DIR / "memery" / "sessions"
SESSION_SUMMARY_DIR = BASE_DIR / "memery" / "session_summaries"


DEFAULT_SOUL = """# soul

你是一个可靠、克制、愿意调用工具解决问题的智能体。

行为准则：
1. 能查就查，不乱猜
2. 有上下文就承接，没有上下文再追问
3. 回答尽量清楚、直接
4. 涉及用户长期偏好时，优先参考 user.md 和 memery.md
"""


DEFAULT_USER = """# user

这里存放用户长期偏好、画像、习惯。
比如：
- 用户偏好中文回答
- 用户常用的模型服务商
- 用户经常关注的公司、城市、任务类型
"""


DEFAULT_MEMERY = """# memery

这里存放跨会话长期记忆摘要。
建议记录：
- 用户长期目标
- 反复提过的项目
- 已经确认过的关键偏好
"""


class MemoryManager:
    """长期记忆管理器。"""

    def __init__(self):
        self._ensure_files()
        self._dedupe_memory_files()

    def _ensure_files(self):
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        if not SOUL_FILE.exists():
            SOUL_FILE.write_text(DEFAULT_SOUL, encoding="utf-8")
        if not USER_FILE.exists():
            USER_FILE.write_text(DEFAULT_USER, encoding="utf-8")
        if not MEMERY_FILE.exists():
            MEMERY_FILE.write_text(DEFAULT_MEMERY, encoding="utf-8")

    def _normalize_memory_text(self, text: str) -> str:
        text = text.strip()
        while text.startswith("-"):
            text = text[1:].strip()
        text = text.replace("：", ":")
        text = "".join(text.split())
        text = text.replace(":", "")
        return text

    def _line_is_memory_item(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith("#"):
            return False
        if stripped.endswith("。") and "：" not in stripped and ":" not in stripped:
            return False
        return True

    def _dedupe_markdown_file(self, file_path: Path):
        if not file_path.exists():
            return

        lines = file_path.read_text(encoding="utf-8").splitlines()
        seen: set[str] = set()
        new_lines: list[str] = []

        for line in lines:
            if not self._line_is_memory_item(line):
                new_lines.append(line)
                continue

            normalized = self._normalize_memory_text(line)
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)

            stripped = line.strip()
            if not stripped.startswith("-"):
                stripped = f"- {stripped.lstrip('-').strip()}"
            new_lines.append(stripped)

        content = "\n".join(new_lines).rstrip() + "\n"
        file_path.write_text(content, encoding="utf-8")

    def _dedupe_memory_files(self):
        self._dedupe_markdown_file(SOUL_FILE)
        self._dedupe_markdown_file(USER_FILE)
        self._dedupe_markdown_file(MEMERY_FILE)

    def _memory_item_exists(self, file_path: Path, content: str) -> bool:
        target = self._normalize_memory_text(content)
        if not target or not file_path.exists():
            return False

        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not self._line_is_memory_item(line):
                continue
            if self._normalize_memory_text(line) == target:
                return True
        return False

    def read_soul(self) -> str:
        return SOUL_FILE.read_text(encoding="utf-8")

    def write_soul(self, content: str):
        SOUL_FILE.write_text(content, encoding="utf-8")

    def read_user_profile(self) -> str:
        return USER_FILE.read_text(encoding="utf-8")

    def write_user_profile(self, content: str):
        USER_FILE.write_text(content, encoding="utf-8")

    def append_user_profile(self, text: str):
        with open(USER_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + text.strip() + "\n")

    def read_long_term_memory(self) -> str:
        return MEMERY_FILE.read_text(encoding="utf-8")

    def write_long_term_memory(self, content: str):
        MEMERY_FILE.write_text(content, encoding="utf-8")

    def append_long_term_memory(self, text: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(MEMERY_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n- [{timestamp}] {text.strip()}\n")

    def update_soul_memory(self, content: str) -> str:
        content = content.strip()
        if not content:
            return "未写入 soul.md：内容为空"
        if self._memory_item_exists(SOUL_FILE, content):
            return f"soul.md 已存在相同记忆，跳过写入：{content}"
        self.write_soul(self.read_soul().rstrip() + "\n- " + content + "\n")
        return f"已写入 soul.md：{content}"

    def update_user_memory(self, content: str) -> str:
        content = content.strip()
        if not content:
            return "未写入 user.md：内容为空"
        if self._memory_item_exists(USER_FILE, content):
            return f"user.md 已存在相同记忆，跳过写入：{content}"
        self.append_user_profile(f"- {content}")
        return f"已写入 user.md：{content}"

    def update_long_term_memory(self, content: str) -> str:
        content = content.strip()
        if not content:
            return "未写入 memery.md：内容为空"
        if self._memory_item_exists(MEMERY_FILE, content):
            return f"memery.md 已存在相同记忆，跳过写入：{content}"
        self.append_long_term_memory(content)
        return f"已写入 memery.md：{content}"

    def build_memory_prompt(self, query: str, top_k: int = 3) -> str:
        """
        为 agent 拼一个可直接放进 prompt 的补充记忆片段。
        """
        return (
            "当前未启用向量检索记忆。"
            "补充记忆主要来自 soul.md、user.md、memery.md 和 session 摘要。"
        )

    def get_session_file(self, session_id: str) -> Path:
        return SESSION_DIR / f"{session_id}.json"

    def load_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        file_path = self.get_session_file(session_id)
        if not file_path.exists():
            return []
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_session_messages(self, session_id: str, messages: list[dict[str, Any]]):
        file_path = self.get_session_file(session_id)
        file_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def append_session_message(self, session_id: str, role: str, content: str):
        messages = self.load_session_messages(session_id)
        messages.append(
            {
                "role": role,
                "content": content,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self.save_session_messages(session_id, messages[-20:])

    def clear_session_messages(self, session_id: str):
        file_path = self.get_session_file(session_id)
        if file_path.exists():
            file_path.unlink()

    def get_session_summary_file(self, session_id: str) -> Path:
        return SESSION_SUMMARY_DIR / f"{session_id}.md"

    def load_session_summary(self, session_id: str) -> str:
        file_path = self.get_session_summary_file(session_id)
        if not file_path.exists():
            return ""
        return file_path.read_text(encoding="utf-8").strip()

    def save_session_summary(self, session_id: str, summary: str):
        file_path = self.get_session_summary_file(session_id)
        file_path.write_text(summary.strip(), encoding="utf-8")

    def clear_session_summary(self, session_id: str):
        file_path = self.get_session_summary_file(session_id)
        if file_path.exists():
            file_path.unlink()

    def get_memory_summary(self) -> dict[str, Any]:
        return {
            "soul_file": str(SOUL_FILE),
            "user_file": str(USER_FILE),
            "memery_file": str(MEMERY_FILE),
            "session_dir": str(SESSION_DIR),
            "session_summary_dir": str(SESSION_SUMMARY_DIR),
            "memory_backend": "markdown_files_only",
        }


if __name__ == "__main__":
    manager = MemoryManager()
    print(manager.get_memory_summary())
    print(manager.read_soul())
