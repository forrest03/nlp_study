"""避免记录用户请求原文的结构化日志工具。"""

import json
import logging
from typing import Any


class StructuredLogFormatter(logging.Formatter):
    """将 Harness 事件格式化为带安全上下文字段的 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        """序列化一条日志记录。

        参数：
            record: 由 Harness 组件发出的日志记录。

        返回：
            包含事件元数据和安全上下文的 JSON 字符串。
        """
        context = getattr(record, "context", {})
        payload: dict[str, Any] = {
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(context)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(verbose: bool) -> None:
    """为 CLI 执行配置包级日志器。

    参数：
        verbose: 是否输出信息级别的生命周期事件。

    返回：
        无。
    """
    logger = logging.getLogger("progressive_harness")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
