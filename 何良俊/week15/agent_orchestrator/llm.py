"""
llm — DeepSeek Chat Completions 客户端（线程安全）
==================================================

编排 Agent 与所有 subagent 共享同一个客户端实例，因此必须线程安全：
- requests 本身线程安全（每次调用独立连接）
- usage 累计计数用锁保护

DeepSeek 兼容 OpenAI Chat Completions 格式：
    POST {base_url}/chat/completions
    Headers: Authorization: Bearer <api_key>
    Body: {"model": "...", "messages": [...], "tools": [...], "tool_choice": "auto"}

依赖：requests
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests


@dataclass
class LLMUsage:
    """累计使用统计（跨线程累计，结束时展示成本）。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def summary(self) -> str:
        return (
            f"调用 {self.calls} 次 | prompt {self.prompt_tokens} tokens | "
            f"completion {self.completion_tokens} tokens | 合计 {self.total_tokens}"
        )


@dataclass
class LLMResponse:
    """一次 chat 调用的统一响应。"""

    text: str = ""                       # 模型文本回复（无 tool_calls 时即最终答案）
    reasoning: str = ""                  # 思考过程（reasoning_content，ReAct 的 Thought）
    message: dict = field(default_factory=dict)   # 完整 assistant 消息，可直接 append 回 messages
    tool_calls: list = field(default_factory=list)  # [{"id","type","function":{name,arguments}}]
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


class DeepSeekClient:
    """DeepSeek Chat Completions 客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-v4-flash",
        timeout: int = 180,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.usage = LLMUsage()
        self._usage_lock = threading.Lock()
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ---- 对外方法 --------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float = 0.0,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        on_tool_call_seen: Optional[Callable[[], None]] = None,
    ) -> LLMResponse:
        """带工具（function calling）的对话调用。

        模型可能直接返回文本回复，也可能返回 tool_calls 让调用方执行后回传。
        stream=True 时文本 token 通过 on_text_delta 实时推送、思考过程通过
        on_reasoning_delta 实时推送、首个 tool_call 增量到达时调用
        on_tool_call_seen()；tool_calls 的增量 delta 会被自动合并，
        流式结束仍返回完整 LLMResponse。
        """
        return self._post(
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "temperature": temperature,
            },
            stream=stream,
            on_text_delta=on_text_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_tool_call_seen=on_tool_call_seen,
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        on_tool_call_seen: Optional[Callable[[], None]] = None,
    ) -> LLMResponse:
        """纯文本对话调用。"""
        return self._post(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
            stream=stream,
            on_text_delta=on_text_delta,
            on_reasoning_delta=on_reasoning_delta,
            on_tool_call_seen=on_tool_call_seen,
        )

    # ---- 内部 ------------------------------------------------------------

    def _post(
        self,
        payload: dict,
        *,
        stream: bool = False,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        on_tool_call_seen: Optional[Callable[[], None]] = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY 环境变量，无法调用 LLM。"
                "请先设置: set DEEPSEEK_API_KEY=sk-xxx（PowerShell 用 $env:）"
            )
        if stream:
            return self._post_stream(
                payload,
                on_text_delta=on_text_delta,
                on_reasoning_delta=on_reasoning_delta,
                on_tool_call_seen=on_tool_call_seen,
            )

        t0 = time.time()
        resp = requests.post(
            self._endpoint, headers=self._headers, json=payload, timeout=self.timeout,
        )
        latency_ms = int((time.time() - t0) * 1000)

        if resp.status_code >= 400:
            raise RuntimeError(f"DeepSeek API 错误 {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}

        with self._usage_lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
            self.usage.completion_tokens += usage.get("completion_tokens", 0)
            self.usage.total_tokens += usage.get("total_tokens", 0)

        return LLMResponse(
            text=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            message=msg,
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason", ""),
            usage=usage,
            latency_ms=latency_ms,
        )

    def _post_stream(
        self,
        payload: dict,
        *,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        on_tool_call_seen: Optional[Callable[[], None]] = None,
    ) -> LLMResponse:
        """流式调用：解析 SSE 事件，实时推送文本/思考 delta，合并 tool_calls 增量。

        DeepSeek/OpenAI 流式格式：每行 `data: {...json...}`，末尾 `data: [DONE]`。
        - 思考增量在 `choices[0].delta.reasoning_content`（通过 on_reasoning_delta 推送）
        - 文本增量在 `choices[0].delta.content`（通过 on_text_delta 推送）
        - 工具调用增量在 `choices[0].delta.tool_calls`（按 index 合并 arguments）
        - usage 在最后一个 chunk（需 stream_options.include_usage）
        """
        payload = {**payload, "stream": True}
        payload["stream_options"] = {"include_usage": True}

        t0 = time.time()
        resp = requests.post(
            self._endpoint, headers=self._headers, json=payload,
            timeout=self.timeout, stream=True,
        )
        latency_ms = int((time.time() - t0) * 1000)

        if resp.status_code >= 400:
            body = resp.text[:500]
            resp.close()
            raise RuntimeError(f"DeepSeek API 错误 {resp.status_code}: {body}")

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        tool_call_seen_fired = False
        finish_reason = ""
        usage: dict = {}

        try:
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].lstrip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # usage 通常在最后一个 chunk（choices 为空时）
                if chunk.get("usage"):
                    usage = chunk["usage"] or {}

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                reasoning_delta = delta.get("reasoning_content")
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    if on_reasoning_delta:
                        on_reasoning_delta(reasoning_delta)

                text_delta = delta.get("content")
                if text_delta:
                    content_parts.append(text_delta)
                    if on_text_delta:
                        on_text_delta(text_delta)

                tc_delta = delta.get("tool_calls")
                if tc_delta:
                    if tc_delta and not tool_call_seen_fired:
                        tool_call_seen_fired = True
                        if on_tool_call_seen:
                            on_tool_call_seen()
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
        reasoning = "".join(reasoning_parts)
        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]

        # 构建可直接 append 回 messages 的完整 assistant 消息
        message: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        with self._usage_lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += usage.get("prompt_tokens", 0)
            self.usage.completion_tokens += usage.get("completion_tokens", 0)
            self.usage.total_tokens += usage.get("total_tokens", 0)

        return LLMResponse(
            text=text,
            reasoning=reasoning,
            message=message,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
        )
