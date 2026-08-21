"""
llm — DeepSeek 客户端封装
============================

DeepSeek API 兼容 OpenAI Chat Completions 格式：
    POST {base_url}/chat/completions
    Headers: Authorization: Bearer <api_key>
    Body: {"model": "...", "messages": [...], "response_format": {...}}

只依赖 `requests`，不引入 openai SDK 以减少依赖体积。

提供三类高层方法：
  - chat(messages, ...)          返回纯文本响应（支持流式）
  - chat_json(messages, ...)     强制 JSON 输出，返回已解析的 dict
  - chat_with_tools(...)         function calling（支持流式）

流式输出：传 stream=True + on_text_delta 回调，文本 token 会实时推送
给回调；tool_calls 的增量 delta 会被自动合并。流式时仍返回完整的
LLMResponse（供记录 / 回传 messages）。

并维护一个简单的 token / 调用计数器，便于在 memory 中记录成本。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from .config import HarnessConfig


@dataclass
class LLMUsage:
    """累计使用统计。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    last_latency_ms: int = 0


@dataclass
class LLMResponse:
    """统一响应对象。"""

    text: str
    json_obj: Optional[dict] = None
    raw: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    # function calling 相关
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""
    message: dict = field(default_factory=dict)  # 完整 assistant message，可直接 append 回 messages


class DeepSeekClient:
    """DeepSeek Chat Completions 客户端。"""

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.usage = LLMUsage()
        self._endpoint = f"{config.deepseek_base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {config.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    # ---- public --------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """普通 chat 调用，返回纯文本。

        stream=True 时通过 on_text_delta 回调实时推送文本 token，
        仍返回完整 LLMResponse。
        """
        return self._call(
            messages,
            response_format=None,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=stream,
            on_text_delta=on_text_delta,
        )

    def chat_json(
        self,
        messages: list[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """强制 JSON 输出。响应对象的 `json_obj` 字段为已解析的 dict。

        DeepSeek 支持 response_format={"type": "json_object"}，但要求 prompt
        中显式提到 'json'（OpenAI 兼容性约束）。不支持流式（需完整解析 JSON）。
        """
        return self._call(
            messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """带工具（function calling）的对话调用。

        DeepSeek 兼容 OpenAI tools 格式。LLM 可直接返回文本回复，也可返回
        tool_calls 让调用方执行后回传结果。响应对象的 `tool_calls` 字段为
        工具调用列表（可能为空），`message` 为可直接 append 回 messages 的
        完整 assistant message。

        stream=True 时通过 on_text_delta 回调实时推送文本 token；
        tool_calls 的增量 delta 会被自动合并。
        """
        return self._call(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            stream=stream,
            on_text_delta=on_text_delta,
        )

    # ---- internal ------------------------------------------------------
    def _call(
        self,
        messages: list[dict],
        *,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        if not self.config.llm_available:
            raise RuntimeError(
                "DEEPSEEK_API_KEY not configured — "
                "set it in environment variable or .env file (see .env.example)"
            )

        payload: dict[str, Any] = {
            "model": self.config.deepseek_model,
            "messages": messages,
            "temperature": self.config.llm_temperature if temperature is None else temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if tools is not None:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        if stream:
            return self._call_stream(payload, on_text_delta=on_text_delta)

        t0 = time.time()
        resp = requests.post(
            self._endpoint,
            headers=self._headers,
            json=payload,
            timeout=self.config.request_timeout,
        )
        latency = int((time.time() - t0) * 1000)

        if resp.status_code >= 400:
            raise RuntimeError(
                f"DeepSeek API error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        finish_reason = choice.get("finish_reason", "")
        usage = data.get("usage", {}) or {}

        # 累计统计
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.usage.completion_tokens += usage.get("completion_tokens", 0)
        self.usage.total_tokens += usage.get("total_tokens", 0)
        self.usage.last_latency_ms = latency

        json_obj = None
        if response_format is not None:
            json_obj = self._safe_parse_json(text)

        return LLMResponse(
            text=text,
            json_obj=json_obj,
            raw=data,
            usage=usage,
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            message=msg,
        )

    def _call_stream(
        self,
        payload: dict[str, Any],
        *,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """流式调用：解析 SSE 事件，实时推送文本 delta，合并 tool_calls 增量。

        DeepSeek/OpenAI 流式格式：每行 `data: {...json...}`，末尾 `data: [DONE]`。
        - 文本增量在 `choices[0].delta.content`
        - 工具调用增量在 `choices[0].delta.tool_calls`（按 index 合并 arguments）
        """
        payload = {**payload, "stream": True}
        # 流式时 DeepSeek 默认不在最后给 usage；开启 stream_options 索要用量
        payload["stream_options"] = {"include_usage": True}

        t0 = time.time()
        resp = requests.post(
            self._endpoint,
            headers=self._headers,
            json=payload,
            timeout=self.config.request_timeout,
            stream=True,
        )
        latency = int((time.time() - t0) * 1000)

        if resp.status_code >= 400:
            body = resp.text[:500]
            resp.close()
            raise RuntimeError(f"DeepSeek API error {resp.status_code}: {body}")

        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        finish_reason = ""
        usage: dict = {}
        raw_chunks: list[dict] = []

        try:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].lstrip()
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                raw_chunks.append(chunk)

                # usage 通常在最后一个 chunk（choices 为空时）
                if chunk.get("usage"):
                    usage = chunk["usage"] or {}

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                # 文本增量
                text_delta = delta.get("content")
                if text_delta:
                    content_parts.append(text_delta)
                    if on_text_delta:
                        on_text_delta(text_delta)

                # tool_calls 增量合并
                tc_delta = delta.get("tool_calls")
                if tc_delta:
                    for tc in tc_delta:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        merged = tool_calls_map[idx]
                        if tc.get("id"):
                            merged["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            merged["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            merged["function"]["arguments"] += fn["arguments"]

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
        finally:
            resp.close()

        text = "".join(content_parts)
        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]

        # 构建 assistant message（可直接 append 回 messages）
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        # 累计统计
        self.usage.calls += 1
        self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
        self.usage.completion_tokens += usage.get("completion_tokens", 0)
        self.usage.total_tokens += usage.get("total_tokens", 0)
        self.usage.last_latency_ms = latency

        return LLMResponse(
            text=text,
            json_obj=None,
            raw={"chunks": raw_chunks},
            usage=usage,
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            message=message,
        )

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[dict]:
        """容错解析：模型可能把 JSON 包在 ```json ... ``` 里。"""
        s = text.strip()
        if s.startswith("```"):
            # 去掉 ```json 或 ``` 围栏
            lines = s.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
