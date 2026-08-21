"""
会话管理模块：支持多轮对话的内存级会话管理与上下文策略

设计要点：
  1. SessionManager 线程安全，使用 threading.Lock 保护所有读写操作
  2. 消息结构兼容 OpenAI 格式：{"role": str, "content": str, ...}
  3. ContextManager 提供按需加载的上下文清理策略（移除 load_skill 中间步骤）

使用方式：
  from session_manager import session_manager, context_manager

  # 创建会话
  session_id = session_manager.create_session()

  # 获取历史消息
  history = session_manager.get_messages(session_id)

  # 保存消息
  session_manager.save_messages(session_id, messages)

  # 应用上下文策略（清理 load_skill 中间步骤）
  compressed = context_manager.apply_strategy(messages, "smart_compact")
"""

import os
import time
import uuid
import logging
import threading
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ── 会话数据结构 ────────────────────────────────────────────────────────────────

class SessionData:
    """单个会话的数据结构"""
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.turn_count: int = 0
        self.token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turn_count": self.turn_count,
            "token_count": self.token_count,
        }


# ── SessionManager：线程安全的会话管理 ──────────────────────────────────────────

class SessionManager:
    """内存级会话管理器，提供线程安全的会话 CRUD 操作"""

    def __init__(self, session_timeout: int = 3600):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._session_timeout = session_timeout

    def create_session(self) -> str:
        """创建新会话，返回会话 ID"""
        session_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._sessions[session_id] = SessionData()
        logger.info(f"创建会话: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionData]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的历史消息列表（返回副本）"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            return [msg.copy() for msg in session.messages]

    def save_messages(self, session_id: str, messages: List[Dict[str, Any]]):
        """保存消息到会话，更新最后活跃时间"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"会话不存在: {session_id}")
                return
            session.messages = messages
            session.last_active = time.time()
            session.token_count = self._estimate_tokens(messages)
            logger.debug(f"保存消息: {session_id}, 消息数: {len(messages)}, token估算: {session.token_count}")

    def increment_turn(self, session_id: str):
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.turn_count += 1

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"删除会话: {session_id}")
                return True
            return False

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def cleanup_expired(self) -> int:
        """清理过期会话，返回清理数量"""
        now = time.time()
        expired_ids = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_active > self._session_timeout:
                    expired_ids.append(session_id)
            for session_id in expired_ids:
                del self._sessions[session_id]
        if expired_ids:
            logger.info(f"清理过期会话: {len(expired_ids)} 个，ID: {expired_ids}")
        return len(expired_ids)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return {
                "session_id": session_id,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "turn_count": session.turn_count,
                "message_count": len(session.messages),
                "token_count": session.token_count,
            }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": sid,
                    "created_at": s.created_at,
                    "last_active": s.last_active,
                    "turn_count": s.turn_count,
                    "message_count": len(s.messages),
                }
                for sid, s in self._sessions.items()
            ]

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """粗略估算 token 数：1 token ≈ 4 字符"""
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4


# ── ContextManager：按需加载的上下文清理策略 ────────────────────────────────────

class ContextManager:
    """
    上下文管理器，配合 skill 按需加载机制清理历史消息

    策略说明：
      smart_compact: 移除 load_skill 的调用与结果（完整定义），保留 execute_skill 的调用与结果。
                     这样上下文中只保留 skill 描述 + 实际执行结果，不保留中间加载的完整定义。
    """

    def __init__(self, max_tokens: int = 8192):
        self._max_tokens = max_tokens

    def apply_strategy(self, messages: List[Dict[str, Any]], strategy: str = "smart_compact") -> List[Dict[str, Any]]:
        if not messages:
            return []

        if strategy == "smart_compact":
            compressed = self._smart_compact(messages)
        else:
            logger.warning(f"未知策略: {strategy}，使用 smart_compact")
            compressed = self._smart_compact(messages)

        logger.debug(f"上下文压缩: 原消息数={len(messages)}, 压缩后={len(compressed)}")
        return compressed

    def _smart_compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按需加载的上下文清理：
          - 移除 load_skill 的 assistant 调用消息和对应的 tool 结果
          - 保留 execute_skill 的 assistant 调用和 tool 结果
          - 保留所有 system / user 消息
        """
        compressed = []
        # 收集需要移除的 tool_call_id（load_skill 调用产生的）
        load_skill_tool_call_ids = set()

        for msg in messages:
            role = msg.get("role")

            # 识别 assistant 消息中的 load_skill tool_call
            if role == "assistant" and msg.get("tool_calls"):
                has_load_skill = False
                for tc in msg.get("tool_calls", []):
                    if tc.get("function", {}).get("name") == "load_skill":
                        load_skill_tool_call_ids.add(tc.get("id"))
                        has_load_skill = True

                if has_load_skill:
                    # 如果该 assistant 消息只有 load_skill 调用（无其他内容），跳过
                    # 如果还有 execute_skill 调用或其他内容，需要保留但移除 load_skill 部分
                    other_calls = [tc for tc in msg.get("tool_calls", []) if tc.get("function", {}).get("name") != "load_skill"]
                    if not other_calls and not msg.get("content"):
                        # 纯 load_skill 调用，整条跳过
                        continue
                    else:
                        # 保留消息但移除 load_skill 的 tool_call
                        msg_copy = msg.copy()
                        msg_copy["tool_calls"] = other_calls
                        compressed.append(msg_copy)
                        continue

            # 跳过 load_skill 对应的 tool 结果
            if role == "tool" and msg.get("tool_call_id") in load_skill_tool_call_ids:
                continue

            compressed.append(msg)

        return compressed

    def get_strategy_info(self) -> Dict[str, Any]:
        return {
            "name": "按需加载清理",
            "description": "移除 load_skill 调用与结果，保留 execute_skill 调用与结果",
            "params": {"max_tokens": self._max_tokens},
        }


# ── 全局单例 ────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TOKENS = int(os.getenv("SESSION_MAX_TOKENS", "8192"))
DEFAULT_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))

session_manager = SessionManager(session_timeout=DEFAULT_TIMEOUT)
context_manager = ContextManager(max_tokens=DEFAULT_MAX_TOKENS)
