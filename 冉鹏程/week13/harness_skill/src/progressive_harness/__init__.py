"""渐进式本地 skill Harness 的公共 API。"""

import logging

from .harness import ProgressiveHarness
from .llm import (
    ChatCompletion,
    LLMConfiguration,
    LLMConfigurationError,
    LLMProtocolError,
    LLMTransportError,
    OpenAICompatibleLLM,
    SkillChatResult,
    SkillEnabledChat,
    ToolCall,
)
from .models import (
    CandidateSkill,
    InvalidReferenceError,
    LoadedReference,
    LoadedSkill,
    SkillMetadata,
    SkillNotFoundError,
)

__all__ = [
    "CandidateSkill",
    "ChatCompletion",
    "InvalidReferenceError",
    "LLMConfiguration",
    "LLMConfigurationError",
    "LLMProtocolError",
    "LLMTransportError",
    "LoadedReference",
    "LoadedSkill",
    "OpenAICompatibleLLM",
    "ProgressiveHarness",
    "SkillChatResult",
    "SkillEnabledChat",
    "SkillMetadata",
    "SkillNotFoundError",
    "ToolCall",
]

logging.getLogger("progressive_harness").addHandler(logging.NullHandler())
