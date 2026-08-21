"""极简 LLM 客户端（蛋糕采集/营销 subagent 用）

默认固定：通义千问 qwen-plus（DashScope OpenAI 兼容接口）。

环境变量：
  DASHSCOPE_API_KEY   必填
  AGENT_MODEL         可选覆盖，默认 qwen-plus
"""
from __future__ import annotations

import os
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"

_client = None
_model = None


def _resolve_provider() -> str:
    return "qwen"


def get_client():
    global _client, _model
    if _client is None:
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise EnvironmentError("请设置 DASHSCOPE_API_KEY（模型固定为 qwen-plus）")
        _client = OpenAI(api_key=key, base_url=QWEN_BASE_URL)
        _model = os.getenv("AGENT_MODEL", DEFAULT_MODEL)
    return _client, _model


def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3):
    """单轮 LLM 对话。stop 用于 ReAct 在 Observation 前截断。"""
    client, model = get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            logger.warning(f"LLM 重试({attempt+1}): {str(e)[:80]}")
