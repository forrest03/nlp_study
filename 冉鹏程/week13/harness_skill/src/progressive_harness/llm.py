"""通过 OpenAI 兼容函数调用接入渐进式 skill 能力。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Any, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .harness import ProgressiveHarness
from .models import CandidateSkill, SkillMetadata

LOGGER = logging.getLogger("progressive_harness.llm")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_HTTP_RESPONSE_BYTES = 2_000_000
MAX_LLM_MESSAGES = 100
MAX_TOOL_ARGUMENT_CHARACTERS = 1_000
MAX_TOOL_CALLS = 6
TOOL_RESULT_ROLE = "tool"


class LLMConfigurationError(ValueError):
    """当 LLM 连接配置缺失或无效时抛出。"""


class LLMTransportError(RuntimeError):
    """当 LLM HTTP 请求无法成功完成时抛出。"""


class LLMProtocolError(RuntimeError):
    """当 LLM 响应或工具调用不符合约定格式时抛出。"""


@dataclass(frozen=True, slots=True)
class LLMConfiguration:
    """描述一个 OpenAI 兼容聊天接口的非敏感连接配置。"""

    api_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """校验连接地址和超时设置，避免发出无效远程请求。"""
        parsed_url = urlparse(self.base_url)
        if not self.api_key or not self.model:
            raise LLMConfigurationError("LLM API Key 和模型名称均不能为空")
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise LLMConfigurationError("LLM_BASE_URL 必须是完整的 HTTP(S) 地址")
        if not 1 <= self.timeout_seconds <= 120:
            raise LLMConfigurationError("LLM 超时必须介于 1 至 120 秒之间")

    @classmethod
    def from_environment(cls) -> "LLMConfiguration":
        """从环境变量构造 LLM 配置。

        参数：
            无。

        返回：
            使用 `LLM_API_KEY`、`LLM_MODEL` 和可选 `LLM_BASE_URL` 的配置。

        异常：
            LLMConfigurationError: 必需变量缺失或值无效时抛出。
        """
        api_key = _required_environment("LLM_API_KEY")
        model = _required_environment("LLM_MODEL")
        base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        return cls(api_key=api_key, model=model, base_url=base_url)

    def endpoint(self) -> str:
        """返回聊天补全请求的完整 URL。

        参数：
            无。

        返回：
            以 `/chat/completions` 结尾的 OpenAI 兼容接口地址。
        """
        normalized_url = self.base_url.rstrip("/")
        if normalized_url.endswith("/chat/completions"):
            return normalized_url
        return f"{normalized_url}/chat/completions"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """表示 LLM 返回的一次函数调用。"""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """表示 LLM 的文本回复和可选函数调用。"""

    content: str | None
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class SkillChatResult:
    """表示一次带 skill 工具调用的 LLM 对话结果。"""

    content: str
    candidates: tuple[CandidateSkill, ...]
    loaded_skill_names: tuple[str, ...]
    loaded_reference_names: tuple[str, ...]
    tool_call_count: int


class ChatCompletionClient(Protocol):
    """定义可供 skill 对话服务使用的 LLM 客户端接口。"""

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ChatCompletion:
        """发送一轮聊天请求并返回结构化的补全结果。

        参数：
            messages: OpenAI 兼容的对话消息。
            tools: 允许模型调用的函数工具定义。

        返回：
            文本内容和函数调用。

        异常：
            LLMTransportError: 远程调用失败时抛出。
            LLMProtocolError: 响应格式无效时抛出。
        """


class OpenAICompatibleLLM:
    """使用标准库访问 OpenAI 兼容 `/chat/completions` 接口。"""

    def __init__(self, configuration: LLMConfiguration) -> None:
        """创建 LLM HTTP 客户端。

        参数：
            configuration: 已校验的非敏感连接配置。

        返回：
            无。
        """
        self._configuration = configuration

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ChatCompletion:
        """调用 LLM 并解析文本回复或工具调用。

        参数：
            messages: OpenAI 兼容的对话消息。
            tools: 允许模型调用的函数工具定义。

        返回：
            解析后的补全结果。

        异常：
            LLMTransportError: 远程调用失败时抛出。
            LLMProtocolError: 输入或响应格式无效时抛出。
        """
        self._validate_messages(messages)
        payload = {"model": self._configuration.model, "messages": list(messages), "tools": list(tools)}
        response_payload = self._post_json(payload)
        return self._completion_from(response_payload)

    def _validate_messages(self, messages: Sequence[dict[str, Any]]) -> None:
        """校验要发送给远程服务的消息数量、角色和文本类型。"""
        if not messages or len(messages) > MAX_LLM_MESSAGES:
            raise LLMProtocolError("LLM 消息数量无效")
        allowed_roles = {"system", "user", "assistant", TOOL_RESULT_ROLE}
        for message in messages:
            is_valid_message = isinstance(message, dict) and message.get("role") in allowed_roles
            is_valid_content = message.get("content") is None or isinstance(message.get("content"), str)
            if not is_valid_message or not is_valid_content:
                raise LLMProtocolError("LLM 消息格式无效")

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 JSON 请求，并以受限大小读取远程响应。"""
        request = Request(
            self._configuration.endpoint(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        LOGGER.info("llm_request_started", extra={"context": {"message_count": len(payload["messages"])}})
        try:
            with urlopen(request, timeout=self._configuration.timeout_seconds) as response:
                response_body = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
        except HTTPError as error:
            LOGGER.warning("llm_request_rejected", extra={"context": {"status_code": error.code}})
            raise LLMTransportError("LLM 服务拒绝了请求") from error
        except (URLError, TimeoutError, OSError) as error:
            LOGGER.warning("llm_request_unavailable", extra={"context": {"error_type": type(error).__name__}})
            raise LLMTransportError("LLM 服务暂时不可用") from error
        return self._decode_response(response_body)

    def _headers(self) -> dict[str, str]:
        """构造请求头，且绝不记录认证信息。"""
        return {
            "Authorization": f"Bearer {self._configuration.api_key}",
            "Content-Type": "application/json",
        }

    def _decode_response(self, response_body: bytes) -> dict[str, Any]:
        """校验远程响应大小并解析为 JSON 对象。"""
        if len(response_body) > MAX_HTTP_RESPONSE_BYTES:
            raise LLMProtocolError("LLM 响应超过允许大小")
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LLMProtocolError("LLM 响应不是有效 JSON") from error
        if not isinstance(payload, dict):
            raise LLMProtocolError("LLM 响应根节点必须是对象")
        return payload

    def _completion_from(self, payload: dict[str, Any]) -> ChatCompletion:
        """从 OpenAI 兼容响应中提取首个候选消息。"""
        try:
            message = payload["choices"][0]["message"]
        except (IndexError, KeyError, TypeError) as error:
            raise LLMProtocolError("LLM 响应缺少候选消息") from error
        if not isinstance(message, dict) or not isinstance(message.get("content"), (str, type(None))):
            raise LLMProtocolError("LLM 返回了无效消息内容")
        return ChatCompletion(content=message.get("content"), tool_calls=self._tool_calls_from(message))

    def _tool_calls_from(self, message: dict[str, Any]) -> tuple[ToolCall, ...]:
        """校验并转换模型返回的函数调用列表。"""
        raw_tool_calls = message.get("tool_calls", [])
        if raw_tool_calls is None:
            return ()
        if not isinstance(raw_tool_calls, list):
            raise LLMProtocolError("LLM 工具调用必须是列表")
        return tuple(self._tool_call_from(item) for item in raw_tool_calls)

    def _tool_call_from(self, item: Any) -> ToolCall:
        """将一个远程工具调用转换为经过校验的数据对象。"""
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            raise LLMProtocolError("LLM 工具调用格式无效")
        function = item["function"]
        call_id = item.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not all(isinstance(value, str) and value for value in (call_id, name, arguments)):
            raise LLMProtocolError("LLM 工具调用字段无效")
        return ToolCall(call_id=call_id, name=name, arguments=arguments)


class SkillEnabledChat:
    """让 LLM 通过受限函数调用逐步使用本地 skill。"""

    def __init__(
        self,
        harness: ProgressiveHarness,
        llm_client: ChatCompletionClient,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        """创建带 skill 工具调用能力的对话服务。

        参数：
            harness: 负责渐进式加载本地 skill 的应用服务。
            llm_client: 支持 OpenAI 风格函数调用的 LLM 客户端。
            max_tool_calls: 单次回答允许的最大工具调用次数。

        返回：
            无。

        异常：
            ValueError: 工具调用次数上限无效时抛出。
        """
        if max_tool_calls < 1:
            raise ValueError("工具调用次数上限必须至少为一")
        self._harness = harness
        self._llm_client = llm_client
        self._max_tool_calls = max_tool_calls

    def answer(self, user_message: str) -> SkillChatResult:
        """回答用户消息，并让 LLM 按需调用本地 skill 工具。

        参数：
            user_message: 未写入日志的用户消息。

        返回：
            最终回答、路由候选项及实际加载的资源清单。

        异常：
            ValueError: 用户消息无效时抛出。
            LLMTransportError: 远程调用失败时抛出。
            LLMProtocolError: LLM 响应或工具调用无效时抛出。
        """
        skills = self._harness.discover()
        candidates = self._harness.select(user_message, skills)
        messages = self._initial_messages(user_message, skills, candidates)
        loaded_skill_names: list[str] = []
        loaded_reference_names: list[str] = []
        return self._run_tool_loop(messages, skills, candidates, loaded_skill_names, loaded_reference_names)

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        skills: tuple[SkillMetadata, ...],
        candidates: tuple[CandidateSkill, ...],
        loaded_skill_names: list[str],
        loaded_reference_names: list[str],
    ) -> SkillChatResult:
        """循环执行模型请求的工具调用，直到获得最终文本。"""
        tool_call_count = 0
        while True:
            completion = self._llm_client.complete(messages, SKILL_TOOLS)
            messages.append(self._assistant_message(completion))
            if not completion.tool_calls:
                return self._result(completion, candidates, loaded_skill_names, loaded_reference_names, tool_call_count)
            if tool_call_count + len(completion.tool_calls) > self._max_tool_calls:
                raise LLMProtocolError("LLM 超过了允许的工具调用次数")
            for tool_call in completion.tool_calls:
                tool_call_count += 1
                result = self._execute_tool(tool_call, skills, loaded_skill_names, loaded_reference_names)
                messages.append(self._tool_message(tool_call.call_id, result))

    def _initial_messages(
        self,
        user_message: str,
        skills: tuple[SkillMetadata, ...],
        candidates: tuple[CandidateSkill, ...],
    ) -> list[dict[str, Any]]:
        """使用元数据目录和路由建议构建初始对话上下文。"""
        return [
            {"role": "system", "content": self._system_prompt(skills, candidates)},
            {"role": "user", "content": user_message},
        ]

    def _system_prompt(
        self,
        skills: tuple[SkillMetadata, ...],
        candidates: tuple[CandidateSkill, ...],
    ) -> str:
        """构建不包含完整 skill 正文的工具使用指令。"""
        catalog = "\n".join(f"- {item.name}: {item.description}" for item in skills)
        suggestions = ", ".join(item.metadata.name for item in candidates) or "无"
        return (
            "你可以使用本地 skill，但目前只获得元数据。\n"
            f"可用 skill：\n{catalog}\n"
            f"基于用户请求的路由建议：{suggestions}。\n"
            "需要使用 skill 时，先调用 load_skill；仅当该说明要求时再调用 load_reference。"
            "不要编造 skill 内容或引用文件内容；不相关时直接回答。"
        )

    def _execute_tool(
        self,
        tool_call: ToolCall,
        skills: tuple[SkillMetadata, ...],
        loaded_skill_names: list[str],
        loaded_reference_names: list[str],
    ) -> dict[str, Any]:
        """执行一个经过验证的 skill 工具调用，并返回可供模型消费的结果。"""
        try:
            return self._successful_tool_result(tool_call, skills, loaded_skill_names, loaded_reference_names)
        except (OSError, ValueError) as error:
            LOGGER.warning(
                "skill_tool_rejected",
                extra={"context": {"tool_name": tool_call.name, "error_type": type(error).__name__}},
            )
            return {"ok": False, "error": type(error).__name__}

    def _successful_tool_result(
        self,
        tool_call: ToolCall,
        skills: tuple[SkillMetadata, ...],
        loaded_skill_names: list[str],
        loaded_reference_names: list[str],
    ) -> dict[str, Any]:
        """处理允许的 load_skill 或 load_reference 调用。"""
        if tool_call.name not in {"load_skill", "load_reference"}:
            raise ValueError("不支持的 skill 工具")
        arguments = _tool_arguments(tool_call.arguments, tool_call.name)
        if tool_call.name == "load_skill":
            return self._load_skill(arguments, skills, loaded_skill_names)
        return self._load_reference(arguments, skills, loaded_skill_names, loaded_reference_names)

    def _load_skill(
        self,
        arguments: dict[str, str],
        skills: tuple[SkillMetadata, ...],
        loaded_skill_names: list[str],
    ) -> dict[str, Any]:
        """加载一个尚未加载的 skill 说明，并避免重复读取。"""
        skill_name = arguments["skill_name"]
        if skill_name in loaded_skill_names:
            return {"ok": True, "stage": "skill", "skill_name": skill_name, "already_loaded": True}
        loaded_skill = self._harness.load_skill(skill_name, skills)
        loaded_skill_names.append(skill_name)
        LOGGER.info("skill_tool_loaded", extra={"context": {"skill_name": skill_name}})
        return {"ok": True, "stage": "skill", "skill_name": skill_name, "instructions": loaded_skill.instructions}

    def _load_reference(
        self,
        arguments: dict[str, str],
        skills: tuple[SkillMetadata, ...],
        loaded_skill_names: list[str],
        loaded_reference_names: list[str],
    ) -> dict[str, Any]:
        """仅在所属 skill 已加载后读取一个明确的引用文件。"""
        skill_name = arguments["skill_name"]
        reference_name = arguments["reference_name"]
        reference_key = f"{skill_name}/{reference_name}"
        if skill_name not in loaded_skill_names:
            raise ValueError("必须先加载 skill 说明后才能读取引用")
        if reference_key in loaded_reference_names:
            return {"ok": True, "stage": "reference", "reference_name": reference_name, "already_loaded": True}
        loaded_reference = self._harness.load_reference(skill_name, reference_name, skills)
        loaded_reference_names.append(reference_key)
        LOGGER.info("reference_tool_loaded", extra={"context": {"skill_name": skill_name, "reference_name": reference_name}})
        return {"ok": True, "stage": "reference", "reference_name": reference_name, "content": loaded_reference.content}

    def _assistant_message(self, completion: ChatCompletion) -> dict[str, Any]:
        """将结构化补全转换为下一轮请求所需的 assistant 消息。"""
        message: dict[str, Any] = {"role": "assistant", "content": completion.content}
        if completion.tool_calls:
            message["tool_calls"] = [
                {"id": item.call_id, "type": "function", "function": {"name": item.name, "arguments": item.arguments}}
                for item in completion.tool_calls
            ]
        return message

    def _tool_message(self, call_id: str, result: dict[str, Any]) -> dict[str, str]:
        """将本地工具结果编码为 OpenAI 兼容的 tool 消息。"""
        return {"role": TOOL_RESULT_ROLE, "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)}

    def _result(
        self,
        completion: ChatCompletion,
        candidates: tuple[CandidateSkill, ...],
        loaded_skill_names: list[str],
        loaded_reference_names: list[str],
        tool_call_count: int,
    ) -> SkillChatResult:
        """验证最终文本并构造不可变对话结果。"""
        if completion.content is None:
            raise LLMProtocolError("LLM 未提供最终回答内容")
        return SkillChatResult(
            content=completion.content,
            candidates=candidates,
            loaded_skill_names=tuple(loaded_skill_names),
            loaded_reference_names=tuple(loaded_reference_names),
            tool_call_count=tool_call_count,
        )


def _required_environment(variable_name: str) -> str:
    """读取必需环境变量，且不会将其值写入日志或错误信息。"""
    value = os.getenv(variable_name)
    if not value:
        raise LLMConfigurationError(f"缺少必需环境变量：{variable_name}")
    return value


def _tool_arguments(arguments: str, tool_name: str) -> dict[str, str]:
    """解析并严格校验模型提供的工具参数对象。"""
    if len(arguments) > MAX_TOOL_ARGUMENT_CHARACTERS:
        raise ValueError("工具参数过长")
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ValueError("工具参数不是有效 JSON") from error
    expected_keys = {"skill_name"} if tool_name == "load_skill" else {"skill_name", "reference_name"}
    if not isinstance(parsed_arguments, dict) or set(parsed_arguments) != expected_keys:
        raise ValueError("工具参数字段无效")
    if not all(isinstance(value, str) and value.strip() for value in parsed_arguments.values()):
        raise ValueError("工具参数必须是非空文本")
    return parsed_arguments


SKILL_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "按需加载一个已列出的本地 skill 的完整 SKILL.md 说明。",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill_name"],
                "properties": {"skill_name": {"type": "string", "minLength": 1}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_reference",
            "description": "在已加载对应 skill 说明后，按需加载其 references 目录中的一个 Markdown 文件。",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill_name", "reference_name"],
                "properties": {
                    "skill_name": {"type": "string", "minLength": 1},
                    "reference_name": {"type": "string", "minLength": 1},
                },
            },
        },
    },
)
