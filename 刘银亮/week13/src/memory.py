"""
记忆管理器

设计要点：
  1. 短期记忆存储在 memory/memory.md，记录用户偏好和基本信息
  2. load_memory() 读取记忆内容，注入 system prompt
  3. update_memory() 通过 LLM 语义提取，将对话中的关键信息合并到 memory.md
  4. 线程安全：使用 threading.Lock 保护文件读写

使用方式：
  from memory import memory_manager

  # 读取记忆（注入 system prompt）
  content = memory_manager.load_memory()

  # 更新记忆（对话结束后调用）
  memory_manager.update_memory(messages)
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
MEMORY_DIR = Path(__file__).parent.parent / "memory"
MEMORY_FILE = MEMORY_DIR / "memory.md"

# ── LLM 客户端（复用 agent 的配置）────────────────────────────────────────────
_client = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
_MODEL = os.getenv("AGENT_MODEL", "qwen-max")

# ── 初始记忆模板 ──────────────────────────────────────────────────────────────
DEFAULT_MEMORY = """# 用户记忆

## 基本信息
- （待补充）

## 偏好
- 偏好中文回答

## 重要事实
- （待补充）
"""

# ── 记忆提取 Prompt ───────────────────────────────────────────────────────────
EXTRACT_PROMPT = """你是记忆提取助手。从以下对话中提取值得长期记忆的信息。

提取范围：
1. 用户偏好（如语言、回答风格、关注领域、技术栈等）
2. 用户基本信息（姓名、职业等，仅当用户明确提及）
3. 重要事实和决策（用户提到的关键信息、明确的要求等）

不要提取：
- 一次性的具体问题（如"123*456等于多少"）
- 工具调用的中间过程
- 无关紧要的闲聊

现有记忆：
{existing_memory}

最近对话（JSON 格式）：
{recent_messages}

请输出更新后的完整记忆内容（Markdown 格式），保留原有信息并补充新信息。
格式必须保持为：
# 用户记忆

## 基本信息
- ...

## 偏好
- ...

## 重要事实
- ...

如果不值得记忆，原样返回现有记忆内容。只输出记忆内容，不要其他解释。"""


class MemoryManager:
    """记忆管理器：读写 memory.md，通过 LLM 进行语义提取"""

    def __init__(self, memory_path: Path = None):
        self._memory_path = memory_path or MEMORY_FILE
        self._lock = threading.Lock()
        self._ensure_memory_file()

    def _ensure_memory_file(self):
        """确保 memory.md 存在"""
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._memory_path.exists():
            self._memory_path.write_text(DEFAULT_MEMORY, encoding="utf-8")
            logger.info(f"初始化记忆文件: {self._memory_path}")

    def load_memory(self) -> str:
        """读取记忆内容"""
        with self._lock:
            try:
                return self._memory_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"读取记忆失败: {e}")
                return DEFAULT_MEMORY

    def save_memory(self, content: str):
        """直接写入记忆内容"""
        with self._lock:
            self._memory_path.write_text(content, encoding="utf-8")
            logger.debug(f"记忆已更新: {self._memory_path}")

    def update_memory(self, messages: List[Dict[str, Any]]):
        """
        通过 LLM 从对话中提取关键信息，更新 memory.md

        Args:
            messages: OpenAI 格式的消息列表
        """
        if not messages:
            return

        # 提取最近几轮对话（避免 token 过多）
        recent = self._extract_recent_dialog(messages)
        if not recent:
            return

        existing = self.load_memory()

        try:
            new_memory = self._llm_extract(existing, recent)
            if new_memory and new_memory.strip():
                self.save_memory(new_memory.strip())
                logger.info("记忆已通过 LLM 更新")
            else:
                logger.debug("LLM 返回空记忆，跳过更新")
        except Exception as e:
            logger.warning(f"LLM 记忆提取失败: {e}")

    def _extract_recent_dialog(self, messages: List[Dict[str, Any]], max_messages: int = 20) -> str:
        """提取最近的对话内容（只保留 user 和 assistant 消息）"""
        dialog = []
        for msg in messages[-max_messages:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                dialog.append({"role": role, "content": content[:500]})  # 截断过长内容
        return json.dumps(dialog, ensure_ascii=False, indent=2)

    def _llm_extract(self, existing_memory: str, recent_dialog: str) -> str:
        """调用 LLM 进行记忆提取"""
        prompt = EXTRACT_PROMPT.format(
            existing_memory=existing_memory,
            recent_messages=recent_dialog,
        )

        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": "你是记忆管理助手，负责维护用户记忆文件。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    def get_memory_info(self) -> Dict[str, Any]:
        """获取记忆文件信息"""
        with self._lock:
            exists = self._memory_path.exists()
            size = self._memory_path.stat().st_size if exists else 0
            return {
                "path": str(self._memory_path),
                "exists": exists,
                "size": size,
                "char_count": len(self.load_memory()) if exists else 0,
            }


# ── 全局单例 ────────────────────────────────────────────────────────────────────
memory_manager = MemoryManager()


if __name__ == "__main__":
    # 测试：读取并打印当前记忆
    print("当前记忆内容:")
    print("=" * 50)
    print(memory_manager.load_memory())
    print("=" * 50)
    print(memory_manager.get_memory_info())
