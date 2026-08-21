import os, time, logging
from openai import OpenAI
logger = logging.getLogger(__name__)

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client_config = OpenAI(
    api_key=os.getenv("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = "deepseek-v4-flash"

_client = None


def get_client():
    global _client
    if _client is None:
        if client_config.api_key:
            _client = OpenAI(api_key=client_config.api_key, base_url=client_config.base_url)
    return _client


def llm_chat(system, user, *, temperature=0.0, max_tokens=1024, stop=None, retries=3,
             model=None):
    """单轮 LLM 对话。stop 用于 ReAct 在 Observation 前截断。"""
    model = model or MODEL
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                temperature=temperature, max_tokens=max_tokens, stop=stop)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries-1: raise
            time.sleep(2**attempt); logger.warning(f"LLM 重试({attempt+1}): {str(e)[:80]}")
