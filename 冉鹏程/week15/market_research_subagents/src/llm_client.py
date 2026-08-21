"""企业信息调查 Agent 使用的极简 DeepSeek LLM 客户端。"""

import logging
import os
import time

from config import load_project_environment

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
_client = None

load_project_environment()


def get_client():
    """延迟创建 OpenAI 兼容客户端。

    返回：配置完成的 DeepSeek 客户端。
    异常：未安装 openai 依赖或未设置 DEEPSEEK_API_KEY 时抛出 EnvironmentError。
    """
    global _client
    if OpenAI is None:
        raise EnvironmentError("缺少 openai 依赖，请执行 pip install -r requirements.txt")
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("请设置 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=api_key, base_url=DEEPSEEK_URL)
    return _client


def llm_chat(system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 1024, stop: list[str] | None = None,
             retries: int = 3) -> str:
    """发起单轮 LLM 对话，并在短暂故障时指数退避重试。

    参数：system 为系统提示，user 为对话历史；stop 用于 ReAct 在 Observation 前截断。
    返回：模型生成的文本。
    异常：重试耗尽后向调用方显式抛出最后一次异常，避免静默产生不完整报告。
    """
    for attempt in range(retries):
        try:
            logger.info("llm_request_started model=%s attempt=%s", DEEPSEEK_MODEL, attempt + 1)
            response = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return response.choices[0].message.content or ""
        except Exception as error:
            if attempt == retries - 1:
                logger.error("llm_request_failed error_type=%s", type(error).__name__)
                raise
            logger.warning("llm_request_retry attempt=%s error_type=%s", attempt + 1, type(error).__name__)
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM 调用未执行")
