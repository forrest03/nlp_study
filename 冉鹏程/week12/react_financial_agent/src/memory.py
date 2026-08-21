"""
会话级短期记忆，管理消息历史与摘要压缩

职责：
  - 维护跨多轮 run() 调用的消息列表，实现多轮对话上下文连续性
  - 当消息 token 数超过阈值时，对早期对话调用 LLM 生成摘要替换原始消息
  - 保留 system prompt + 摘要 + 近期完整对话，确保消息前缀稳定（KV Cache 友好）

公开类：
  ShortTermMemory - 会话级短期记忆管理器

公开方法：
  add_message(role, content, **kwargs) -> None
    追加一条消息到历史，若超过阈值则触发压缩
    kwargs 用于传递 FC 版的 tool_calls / tool_call_id 等额外字段

  get_messages() -> list[dict]
    返回内部消息列表引用（非拷贝），调用方可直接 append 修改

  clear() -> None
    清空当前会话记忆，仅保留 system prompt

  touch() -> None
    更新最后活跃时间戳，用于会话过期清理

  last_active_at -> float
    最后活跃时间的 Unix 时间戳
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_MODEL = os.getenv("AGENT_MODEL", "qwen-max")

# 懒加载 LLM 客户端，避免 import 时就要求 API key
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """获取或创建 LLM 客户端实例（懒加载）"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _client

# 摘要生成的 Prompt
_SUMMARY_PROMPT = (
    "请用简洁中文总结以下对话的关键信息和结论，"
    "保留所有重要的数据、数字、公司名称和股票代码，"
    "省略推理过程和工具调用细节，只保留最终结论和关键事实。"
)

# 中文约 1.5 字/token，英文约 4 字符/token，取加权平均
_CHARS_PER_TOKEN = 2.0


class ShortTermMemory:
    """会话级短期记忆，管理消息历史与摘要压缩

    压缩策略：当估算 token 数超过 max_tokens * compress_threshold 时，
    将早期对话交给 LLM 生成摘要，替换为一条 system 角色的摘要消息。
    保留最近 _KEEP_RECENT_ROUNDS 轮完整对话不动，确保前缀稳定。
    """

    _KEEP_RECENT_ROUNDS = 2  # 压缩时保留最近几轮完整对话

    def __init__(
        self,
        system_prompt: str,
        max_tokens: int = 6000,
        compress_threshold: float = 0.8,
    ):
        """
        Args:
            system_prompt: 系统提示词，始终保留在消息列表头部
            max_tokens: 消息列表的 token 上限
            compress_threshold: 触发压缩的阈值比例（0.8 = 80%时触发）
        """
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._compress_threshold = compress_threshold
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        self._summary: Optional[str] = None
        self.last_active_at: float = time.time()

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """追加一条消息到历史，若超过阈值则触发压缩

        Args:
            role: 消息角色（system/user/assistant/tool）
            content: 消息内容
            **kwargs: 额外字段，如 tool_calls、tool_call_id 等
        """
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(kwargs)
        self._messages.append(msg)
        self.touch()

        if self._should_compress():
            self._compress()

    def get_messages(self) -> list[dict[str, Any]]:
        """返回内部消息列表引用（非拷贝），调用方可直接 append 修改

        Why 返回引用：ReAct 循环内直接操作 messages 列表（append），
        返回引用避免每次循环都需同步回 memory，压缩时直接修改 _messages。
        """
        return self._messages

    def clear(self) -> None:
        """清空当前会话记忆，仅保留 system prompt"""
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._summary = None
        self.touch()
        logger.info("会话记忆已清空")

    def touch(self) -> None:
        """更新最后活跃时间戳，用于会话过期清理"""
        self.last_active_at = time.time()

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _should_compress(self) -> bool:
        """判断是否需要触发压缩"""
        estimated = self._estimate_tokens(self._messages)
        threshold = int(self._max_tokens * self._compress_threshold)
        return estimated > threshold

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """估算消息列表的 token 数

        启发式：中文约 1.5 字/token，英文约 4 字符/token，
        取加权平均 _CHARS_PER_TOKEN，加上每条消息的固定开销。
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # FC 版 content 可能是 list（多模态），取文本部分
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
        # 每条消息额外开销约 4 token（role 标记、分隔符等）
        overhead = len(messages) * 4
        return int(total_chars / _CHARS_PER_TOKEN) + overhead

    def _compress(self) -> None:
        """压缩早期对话：将较早的消息交给 LLM 生成摘要，替换原始消息

        流程：
          1. 分离 system_prompt | 早期消息 | 近期完整对话
          2. 对早期消息调用 LLM 生成摘要
          3. 用摘要消息替换早期消息
          4. 最终结构：[system_prompt] + [摘要] + [近期完整对话]
        """
        # 找到 system prompt 之后、最近 _KEEP_RECENT_ROUNDS 轮之前的位置
        split_idx = self._find_compress_boundary()
        if split_idx <= 1:
            # 早期消息太少，无需压缩
            return

        early_messages = self._messages[1:split_idx]
        recent_messages = self._messages[split_idx:]

        summary_text = self._generate_summary(early_messages)
        if not summary_text:
            logger.warning("摘要生成失败，跳过本次压缩")
            return

        # 合并已有摘要（如果之前已压缩过）
        if self._summary:
            summary_text = f"{self._summary}\n\n--- 后续对话补充 ---\n{summary_text}"
        self._summary = summary_text

        summary_msg = {
            "role": "system",
            "content": f"[历史对话摘要]\n{summary_text}",
        }
        self._messages = [self._messages[0], summary_msg] + recent_messages
        logger.info(
            "记忆压缩完成：早期 %d 条消息 → 1 条摘要，"
            "当前总消息数 %d，估算 token %d",
            len(early_messages),
            len(self._messages),
            self._estimate_tokens(self._messages),
        )

    def _find_compress_boundary(self) -> int:
        """找到压缩边界索引：保留最近 _KEEP_RECENT_ROUNDS 轮完整对话

        一轮 = user 消息 + 对应的 assistant/tool 消息序列。
        从后往前数 _KEEP_RECENT_ROUNDS 个 user 消息，返回第一个的位置。
        """
        user_count = 0
        for i in range(len(self._messages) - 1, 0, -1):
            if self._messages[i].get("role") == "user":
                user_count += 1
                if user_count >= self._KEEP_RECENT_ROUNDS:
                    return i
        # user 消息不足 _KEEP_RECENT_ROUNDS 轮，不压缩
        return 1

    def _generate_summary(self, messages: list[dict[str, Any]]) -> str:
        """调用 LLM 对早期消息生成摘要

        Args:
            messages: 需要摘要的早期消息列表

        Returns:
            摘要文本，失败时返回空字符串
        """
        try:
            summary_messages = [
                {"role": "system", "content": _SUMMARY_PROMPT},
                {"role": "user", "content": self._format_for_summary(messages)},
            ]
            resp = _get_client().chat.completions.create(
                model=_MODEL,
                messages=summary_messages,
                temperature=0,
                max_tokens=500,
            )
            result = resp.choices[0].message.content.strip()
            logger.info("摘要生成成功，长度 %d 字", len(result))
            return result
        except Exception as e:
            logger.error("摘要生成失败: %s", e, exc_info=True)
            return ""

    @staticmethod
    def _format_for_summary(messages: list[dict[str, Any]]) -> str:
        """将消息列表格式化为摘要用的纯文本

        过滤 tool 角色的冗长原始输出，只保留关键信息。
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            # 截断过长的 observation，保留前 200 字
            if role == "user" and content.startswith("Observation:"):
                content = content[:200] + ("..." if len(content) > 200 else "")
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)
