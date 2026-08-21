"""
memory.py — 记忆策略

把"喂给 LLM 的 messages 列表"这件事抽象成策略接口，
未来加滑动窗口 / 摘要压缩时只新增策略类，Runner 无感。

当前实现：FullHistoryMemory（完整历史直通，YAGNI 起手）
"""

from typing import Protocol


class MemoryStrategy(Protocol):
    def prepare(self, messages: list[dict]) -> list[dict]: ...


class FullHistoryMemory:
    """最朴素的策略：原样返回。"""

    def prepare(self, messages: list[dict]) -> list[dict]:
        return list(messages)
