"""
会话管理模块：支持多轮对话的内存级会话管理与上下文策略

设计要点：
  1. SessionManager 线程安全，使用 threading.Lock 保护所有读写操作
  2. 消息结构兼容 OpenAI 格式：{"role": str, "content": str, ...}
  3. 每个会话存储完整的 ReAct 轮次（Thought/Action/Observation），作为逻辑单元
  4. ContextManager 提供多种策略：滑动窗口、Token截断、智能压缩

使用方式：
  from session_manager import session_manager, context_manager

  # 创建会话
  session_id = session_manager.create_session()

  # 获取历史消息
  history = session_manager.get_messages(session_id)

  # 保存消息（每次 ReAct 轮次完成后）
  session_manager.save_messages(session_id, messages)

  # 应用上下文策略
  compressed = context_manager.apply_strategy(messages, "smart_compact")
"""

import os
import json
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
        self.messages: List[Dict[str, Any]] = []  # OpenAI 格式消息列表
        self.created_at: float = time.time()      # 创建时间戳
        self.last_active: float = time.time()      # 最后活跃时间戳
        self.turn_count: int = 0                   # 用户提问轮次计数
        self.token_count: int = 0                  # 累计 token 消耗估算

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，便于外部使用"""
        return {
            "messages": self.messages,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "turn_count": self.turn_count,
            "token_count": self.token_count,
        }


# ── SessionManager：线程安全的会话管理 ──────────────────────────────────────────

class SessionManager:
    """
    内存级会话管理器，提供线程安全的会话 CRUD 操作
    """
    def __init__(self, session_timeout: int = 3600):
        """
        Args:
            session_timeout: 会话超时时间（秒），默认1小时
        """
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._session_timeout = session_timeout

    def create_session(self) -> str:
        """创建新会话，返回会话 ID"""
        session_id = uuid.uuid4().hex[:8]  # 8位短ID，便于前端传递
        with self._lock:
            self._sessions[session_id] = SessionData()
        logger.info(f"创建会话: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话数据（内部使用）"""
        with self._lock:
            return self._sessions.get(session_id)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的历史消息列表"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            # 返回副本，防止外部修改
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
        """增加用户提问轮次计数"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.turn_count += 1

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"删除会话: {session_id}")
                return True
            return False

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
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
        """获取会话元信息（不含消息）"""
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
        """列出所有活跃会话的摘要信息"""
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
        """估算消息列表的 token 数量（粗略估算：1 token ≈ 4 字符）"""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4


# ── ContextManager：多种上下文管理策略 ──────────────────────────────────────────

class ContextManager:
    """
    上下文管理器，提供多种策略压缩/截断对话历史

    策略说明：
      1. sliding_window: 保留最近 N 轮用户提问及相关回复
      2. token_truncate: 保留在最大 token 数内，从后往前截断
      3. smart_compact: 智能压缩，保留系统提示 + 工具结果 + 最终答案，丢弃中间思考
    """

    def __init__(self, max_tokens: int = 8192, window_size: int = 5):
        """
        Args:
            max_tokens: 默认最大 token 限制
            window_size: 默认滑动窗口大小
        """
        self._max_tokens = max_tokens
        self._window_size = window_size

    def apply_strategy(self, messages: List[Dict[str, Any]], strategy: str = "smart_compact") -> List[Dict[str, Any]]:
        """
        应用指定的上下文管理策略

        Args:
            messages: 原始消息列表
            strategy: 策略名称，可选: "sliding_window", "token_truncate", "smart_compact"

        Returns:
            压缩后的消息列表
        """
        if not messages:
            return []

        strategy_map = {
            "sliding_window": self._sliding_window,
            "token_truncate": self._token_truncate,
            "smart_compact": self._smart_compact,
        }

        fn = strategy_map.get(strategy)
        if not fn:
            logger.warning(f"未知策略: {strategy}，使用 smart_compact")
            fn = self._smart_compact

        compressed = fn(messages)
        logger.debug(f"上下文压缩: 原消息数={len(messages)}, 压缩后={len(compressed)}, 策略={strategy}")
        return compressed

    def _sliding_window(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        滑动窗口策略：保留最近 N 轮完整对话

        轮次定义：用户提问 + 后续所有 ReAct 步骤（Thought/Action/Observation）+ 最终答案
        """
        # 提取系统提示词
        system_msgs = [m for m in messages if m.get("role") == "system"]

        # 找出所有用户提问的索引
        user_indices = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                # 排除 Observation 开头的用户消息（ReAct 内部步骤）
                content = msg.get("content", "")
                if not content.strip().startswith("Observation:"):
                    user_indices.append(i)

        if not user_indices:
            return messages

        # 保留最近 window_size 个用户提问及其后续内容
        start_idx = user_indices[-self._window_size] if len(user_indices) >= self._window_size else user_indices[0]
        recent_msgs = messages[start_idx:]

        return system_msgs + recent_msgs

    def _token_truncate(self, messages: List[Dict[str, Any]], max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Token 截断策略：从后往前保留，确保总 token 数不超过限制

        始终保留系统提示词，从后往前计算 token 数
        """
        max_tokens = max_tokens or self._max_tokens

        # 提取系统提示词
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]

        # 计算系统提示词的 token 数
        system_tokens = sum(len(m.get("content", "")) // 4 for m in system_msgs)

        if system_tokens >= max_tokens:
            logger.warning(f"系统提示词已超出 token 限制: {system_tokens} > {max_tokens}")
            return system_msgs[:1]  # 只保留第一个系统提示

        remaining_tokens = max_tokens - system_tokens

        # 从后往前计算，保留尽可能多的消息
        truncated = []
        current_tokens = 0

        for msg in reversed(non_system_msgs):
            msg_tokens = len(msg.get("content", "")) // 4
            if current_tokens + msg_tokens > remaining_tokens:
                # 如果当前消息会超限制，尝试截断内容
                if remaining_tokens > 0:
                    max_chars = remaining_tokens * 4
                    truncated_content = msg.get("content", "")[:max_chars] + "..."
                    truncated.insert(0, {**msg, "content": truncated_content})
                break
            truncated.insert(0, msg)
            current_tokens += msg_tokens

        return system_msgs + truncated

    def _smart_compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        智能压缩策略：
          - 保留系统提示词（始终）
          - 保留用户提问（所有）
          - 保留工具调用结果（role=tool 或 Observation 内容）
          - 保留最终答案（包含 Final Answer 的 assistant 消息）
          - 丢弃中间思考过程（纯 Thought 的 assistant 消息）

        此策略特别适合 ReAct 模式，减少无效的中间推理步骤带来的 token 浪费
        """
        system_msgs = []
        user_msgs = []
        tool_results = []
        final_answers = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_msgs.append(msg)

            elif role == "user":
                # 区分用户原始提问和 ReAct 内部的 Observation
                if content.strip().startswith("Observation:"):
                    # 这是工具执行结果的包装，归入工具结果
                    tool_results.append(msg)
                else:
                    user_msgs.append(msg)

            elif role == "tool":
                # Function Calling 格式的工具结果
                tool_results.append(msg)

            elif role == "assistant":
                # 判断是否包含最终答案
                if "Final Answer:" in content:
                    final_answers.append(msg)
                # 否则丢弃中间思考过程（Thought）

        # 按时间顺序重组：系统提示 + 用户提问 + 工具结果 + 最终答案
        # 注意：这里简化处理，实际应用中可能需要更复杂的顺序保持
        # 为保持上下文连贯性，我们保留原始消息结构，但过滤掉中间思考
        compact = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                compact.append(msg)

            elif role == "user":
                compact.append(msg)

            elif role == "tool":
                compact.append(msg)

            elif role == "assistant":
                # 只保留包含最终答案的 assistant 消息
                if "Final Answer:" in content:
                    compact.append(msg)

        return compact

    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """获取策略的描述信息"""
        info = {
            "sliding_window": {
                "name": "滑动窗口",
                "description": "保留最近 N 轮完整对话，适用于大多数场景",
                "params": {"window_size": self._window_size},
            },
            "token_truncate": {
                "name": "Token 截断",
                "description": "从后往前截断，确保总 token 数不超过限制",
                "params": {"max_tokens": self._max_tokens},
            },
            "smart_compact": {
                "name": "智能压缩",
                "description": "针对 ReAct 模式优化，保留系统提示、用户提问、工具结果和最终答案，丢弃中间思考",
                "params": {},
            },
        }
        return info.get(strategy, {"name": "未知策略", "description": "", "params": {}})


# ── 全局单例 ────────────────────────────────────────────────────────────────────

# 从环境变量读取配置
DEFAULT_MAX_TOKENS = int(os.getenv("SESSION_MAX_TOKENS", "8192"))
DEFAULT_WINDOW_SIZE = int(os.getenv("SESSION_WINDOW_SIZE", "5"))
DEFAULT_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))

# 创建全局单例，便于其他模块直接使用
session_manager = SessionManager(session_timeout=DEFAULT_TIMEOUT)
context_manager = ContextManager(max_tokens=DEFAULT_MAX_TOKENS, window_size=DEFAULT_WINDOW_SIZE)
