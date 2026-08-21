"""极简 LLM 客户端（股票 subagent 项目用）

按用户偏好使用 DASHSCOPE_API_KEY 调用 Qwen 模型（阿里云百炼），
走 OpenAI 兼容接口（DashScope compatible-mode），无需额外 SDK。

依赖：pip install openai
"""
import os, time, logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# DashScope OpenAI 兼容端点
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 默认模型（可由环境变量覆盖，按需切换 qwen-turbo / qwen-plus / qwen-max）
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise EnvironmentError("请设置 DASHSCOPE_API_KEY（阿里云百炼）")
        _client = OpenAI(api_key=key, base_url=DASHSCOPE_URL)
    return _client


def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3):
    """单轮 LLM 对话。stop 用于 ReAct 在 Observation 前截断。"""
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            logger.warning(f"LLM 重试({attempt + 1}): {str(e)[:80]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(llm_chat("你是测试助手", "说一句你好", max_tokens=64))
