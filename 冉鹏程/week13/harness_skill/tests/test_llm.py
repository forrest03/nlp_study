"""覆盖 LLM skill 工具调用、边界和配置异常的单元测试。"""

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from progressive_harness import (
    ChatCompletion,
    LLMConfiguration,
    LLMConfigurationError,
    ProgressiveHarness,
    SkillEnabledChat,
    ToolCall,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = PROJECT_ROOT / "skills"


class ScriptedLLM:
    """按预设顺序返回补全结果的无网络 LLM 测试替身。"""

    def __init__(self, responses: list[ChatCompletion]) -> None:
        """创建一个带固定响应序列的测试替身。

        参数：
            responses: 每次调用 `complete` 依次返回的补全结果。

        返回：
            无。
        """
        self._responses = list(responses)
        self.requests: list[tuple[list[dict], list[dict]]] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatCompletion:
        """记录请求并返回下一个预设结果。

        参数：
            messages: 当前发给模型的消息。
            tools: 当前允许调用的工具定义。

        返回：
            下一个预设的补全结果。

        异常：
            AssertionError: 预设响应已经耗尽时抛出。
        """
        self.requests.append((deepcopy(messages), deepcopy(tools)))
        if not self._responses:
            raise AssertionError("测试 LLM 的预设响应已耗尽")
        return self._responses.pop(0)


class SkillEnabledChatTests(unittest.TestCase):
    """验证 LLM 只能通过工具按阶段加载本地 skill。"""

    def setUp(self) -> None:
        """为每个测试创建独立的本地 Harness。

        参数：
            无。

        返回：
            无。
        """
        self.harness = ProgressiveHarness(SKILLS_ROOT)

    def test_tool_loop_loads_skill_then_reference(self) -> None:
        """模型可按 skill 再引用的顺序获得所需上下文。

        参数：
            无。

        返回：
            无。
        """
        llm = ScriptedLLM([
            ChatCompletion(None, (ToolCall("skill-1", "load_skill", '{"skill_name":"baoyu-diagram"}'),)),
            ChatCompletion(None, (ToolCall("reference-1", "load_reference", '{"skill_name":"baoyu-diagram","reference_name":"architecture.md"}'),)),
            ChatCompletion("已加载架构图 skill 后生成回答。", ()),
        ])

        result = SkillEnabledChat(self.harness, llm).answer("画一个订单系统架构图")

        self.assertEqual(result.content, "已加载架构图 skill 后生成回答。")
        self.assertEqual(result.loaded_skill_names, ("baoyu-diagram",))
        self.assertEqual(result.loaded_reference_names, ("baoyu-diagram/architecture.md",))
        self.assertEqual(result.tool_call_count, 2)
        self.assertIn("load_skill", {item["function"]["name"] for item in llm.requests[0][1]})
        self.assertIn("# 图表生成器", llm.requests[1][0][-1]["content"])
        self.assertIn("架构图", llm.requests[2][0][-1]["content"])

    def test_unsupported_tool_is_reported_without_loading_files(self) -> None:
        """不支持的模型工具调用会返回错误结果且不读取本地文件。

        参数：
            无。

        返回：
            无。
        """
        llm = ScriptedLLM([
            ChatCompletion(None, (ToolCall("invalid-1", "delete_skill", "{}"),)),
            ChatCompletion("已忽略无效工具。", ()),
        ])

        result = SkillEnabledChat(self.harness, llm).answer("随便回答")

        tool_result = json.loads(llm.requests[1][0][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertEqual(result.loaded_skill_names, ())
        self.assertEqual(result.loaded_reference_names, ())

    def test_reference_requires_prior_skill_load(self) -> None:
        """模型不能在未读取 skill 说明前直接获取引用文件。

        参数：
            无。

        返回：
            无。
        """
        llm = ScriptedLLM([
            ChatCompletion(None, (ToolCall("reference-first", "load_reference", '{"skill_name":"baoyu-diagram","reference_name":"architecture.md"}'),)),
            ChatCompletion("引用需要先加载 skill。", ()),
        ])

        result = SkillEnabledChat(self.harness, llm).answer("画架构图")

        tool_result = json.loads(llm.requests[1][0][-1]["content"])
        self.assertFalse(tool_result["ok"])
        self.assertEqual(result.loaded_reference_names, ())

    def test_final_answer_after_maximum_allowed_call_is_accepted(self) -> None:
        """达到工具调用上限后，只要返回最终文本仍应视为成功。

        参数：
            无。

        返回：
            无。
        """
        llm = ScriptedLLM([
            ChatCompletion(None, (ToolCall("skill-only", "load_skill", '{"skill_name":"baoyu-diagram"}'),)),
            ChatCompletion("已使用一个 skill。", ()),
        ])

        result = SkillEnabledChat(self.harness, llm, max_tool_calls=1).answer("画架构图")

        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.content, "已使用一个 skill。")

    def test_environment_configuration_requires_key_and_model(self) -> None:
        """环境变量缺少密钥或模型时会显式失败。

        参数：
            无。

        返回：
            无。
        """
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMConfigurationError):
                LLMConfiguration.from_environment()
