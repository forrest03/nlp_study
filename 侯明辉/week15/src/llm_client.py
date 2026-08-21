"""
硅基流动 LLM 客户端（OpenAI 兼容）。

提供单一函数 llm_chat(system, user) -> str。
支持 response_format_json=True 走 json_object 模式。
失败指数退避重试 3 次。

依赖：openai>=1.0.0
环境变量：SILICONFLOW_API_KEY
"""
import os
import time
import logging

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "请设置环境变量 SILICONFLOW_API_KEY（复制 .env.example 为 .env 再 source）"
            )
        from openai import OpenAI
        base_url = os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def llm_chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_format_json: bool = False,
    stop: list[str] | None = None,
    retries: int = 3,
) -> str:
    """
    单轮 LLM 对话，返回文本。失败自动重试（指数退避）。

    参数:
      response_format_json: True 时走 json_object 模式（硅基流动兼容 OpenAI）
      stop: 可选的 stop token 列表（如 ["Observation:"] 让 LLM 在该位置停）
    """
    client = _get_client()
    model = os.getenv("SILICONFLOW_MODEL", DEFAULT_MODEL)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    if stop is not None:
        kwargs["stop"] = stop

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"LLM 调用失败({type(e).__name__})，{wait}s 后重试: {str(e)[:80]}")
            time.sleep(wait)


if __name__ == "__main__":
    # 自测：调一次 LLM，确认 API 通
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    out = llm_chat("你是一个测试助手", "只回复 OK 两个字", max_tokens=10)
    print(f"LLM 自测: {out!r}")
    assert "OK" in out, f"期望 OK，实际 {out!r}"
    print("✓ llm_client 自测通过")
