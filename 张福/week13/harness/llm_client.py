"""
LLM 客户端模块

从环境变量 DASHSCOPE_API_KEY 读取 API Key，调用 DashScope 兼容 OpenAI 接口。
支持原生 OpenAI function calling（工具调用）。
"""

import json
import os
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = os.getenv("HARNESS_MODEL", "qwen-plus")


def get_client(api_key: Optional[str] = None) -> OpenAI:
    """获取 DashScope OpenAI 兼容客户端。"""
    key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise EnvironmentError(
            "请设置环境变量 DASHSCOPE_API_KEY\n"
            "  export DASHSCOPE_API_KEY=sk-xxx"
        )
    return OpenAI(api_key=key, base_url=DASHSCOPE_URL)


def chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> str:
    """非流式对话，返回 assistant 回复文本。"""
    client = get_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def chat_stream(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> Generator[str, None, None]:
    """流式对话，逐块 yield 文本。"""
    client = get_client(api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def chat_with_tools(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> tuple[str, List[Any]]:
    """非流式对话（支持工具调用）。

    Returns:
        (assistant_text_content, list_of_tool_calls)
        如果 LLM 返回了工具调用则 tool_calls 非空；否则为 []。
    """
    client = get_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
    )
    msg = resp.choices[0].message
    return (msg.content or "", msg.tool_calls or [])
