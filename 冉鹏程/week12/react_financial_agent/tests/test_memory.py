"""
ShortTermMemory 单元测试

覆盖场景：
  1. 新建 memory，添加消息，get_messages 返回正确
  2. Token 估算基本准确性
  3. 压缩触发（mock LLM 调用）
  4. 压缩后消息结构：system_prompt + 摘要 + 近期对话
  5. 多次压缩时摘要合并
  6. 空会话 clear
  7. FC 版 tool_calls / tool_call_id 字段保留
  8. touch 更新活跃时间
  9. _find_compress_boundary 边界情况
"""

import os
import sys
import time
import pytest
from unittest.mock import patch, MagicMock

# 确保 src 在搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory import ShortTermMemory


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def memory():
    """创建一个低 token 阈值的 memory 实例，便于触发压缩测试"""
    return ShortTermMemory(
        system_prompt="你是测试助手",
        max_tokens=200,
        compress_threshold=0.8,
    )


# ── 基础功能测试 ────────────────────────────────────────────────────────────

class TestBasicOperations:
    """测试消息添加、获取、清空等基础操作"""

    def test_new_memory_has_system_prompt(self, memory):
        msgs = memory.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "你是测试助手"

    def test_add_user_message(self, memory):
        memory.add_message("user", "茅台2023年毛利率是多少？")
        msgs = memory.get_messages()
        assert len(msgs) == 2
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "茅台2023年毛利率是多少？"

    def test_add_multiple_messages(self, memory):
        memory.add_message("user", "问题1")
        memory.add_message("assistant", "回答1")
        memory.add_message("user", "问题2")
        assert len(memory.get_messages()) == 4

    def test_get_messages_returns_reference(self, memory):
        """get_messages 返回引用，append 直接影响内部列表"""
        msgs = memory.get_messages()
        msgs.append({"role": "user", "content": "直接追加"})
        assert len(memory.get_messages()) == 2

    def test_clear_resets_to_system_prompt(self, memory):
        memory.add_message("user", "问题")
        memory.add_message("assistant", "回答")
        memory.clear()
        msgs = memory.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_add_message_with_kwargs(self, memory):
        """FC 版的 tool_calls / tool_call_id 通过 kwargs 传递"""
        memory.add_message("assistant", None, tool_calls=[{"id": "call_1"}])
        memory.add_message("tool", "结果", tool_call_id="call_1")
        msgs = memory.get_messages()
        assert msgs[1].get("tool_calls") == [{"id": "call_1"}]
        assert msgs[2].get("tool_call_id") == "call_1"


# ── Token 估算测试 ──────────────────────────────────────────────────────────

class TestTokenEstimation:
    """测试 token 估算逻辑"""

    def test_estimate_tokens_basic(self, memory):
        msgs = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
        ]
        tokens = memory._estimate_tokens(msgs)
        assert tokens > 0
        # 2 条消息 * 4 overhead + (3+2)/2.0 ≈ 10 + 2.5 = 12
        assert 5 < tokens < 30

    def test_estimate_tokens_empty_content(self, memory):
        msgs = [{"role": "system", "content": ""}]
        tokens = memory._estimate_tokens(msgs)
        assert tokens == 4  # 仅 overhead

    def test_estimate_tokens_list_content(self, memory):
        """FC 版 content 可能是 list（多模态）"""
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "描述图片"},
                {"type": "image_url", "image_url": {"url": "http://x"}},
            ]},
        ]
        tokens = memory._estimate_tokens(msgs)
        assert tokens > 4  # overhead + 文本部分


# ── 压缩测试 ────────────────────────────────────────────────────────────────

class TestCompression:
    """测试摘要压缩逻辑"""

    @patch("memory._get_client")
    def test_compress_triggered_by_token_threshold(self, mock_get_client, memory):
        """当 token 超过阈值时自动触发压缩"""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "茅台毛利率91.96%"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        # 添加足够多的消息触发压缩（max_tokens=200, threshold=0.8 → 160 token 触发）
        for i in range(10):
            memory.add_message("user", f"第{i+1}个问题，这是一段较长的中文文本内容" * 5)
            memory.add_message("assistant", f"第{i+1}个回答，这是一段较长的中文文本内容" * 5)

        # 应该调用了 LLM 生成摘要
        mock_client.chat.completions.create.assert_called()

    @patch("memory._get_client")
    def test_compress_preserves_recent_conversation(self, mock_get_client, memory):
        """压缩后保留最近 2 轮完整对话"""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "早期对话摘要内容"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        # 添加 5 轮对话
        for i in range(5):
            memory.add_message("user", f"问题{i}" * 20)
            memory.add_message("assistant", f"回答{i}" * 20)

        msgs = memory.get_messages()
        # 应该有：system + 摘要 + 最近 2 轮（4 条消息）
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "你是测试助手"
        assert msgs[1]["role"] == "system"
        assert "[历史对话摘要]" in msgs[1]["content"]
        # 最近 2 轮保留
        recent_contents = [m["content"] for m in msgs[2:]]
        assert any("问题3" in c for c in recent_contents)
        assert any("问题4" in c for c in recent_contents)

    @patch("memory._get_client")
    def test_multiple_compress_merges_summaries(self, mock_get_client, memory):
        """多次压缩时，摘要应合并而非覆盖"""
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = f"第{call_count}次摘要"
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = mock_create
        mock_get_client.return_value = mock_client

        # 第一轮：添加足够消息触发第一次压缩
        for i in range(6):
            memory.add_message("user", f"第一轮问题{i}" * 20)
            memory.add_message("assistant", f"第一轮回答{i}" * 20)

        # 第二轮：继续添加触发第二次压缩
        for i in range(6):
            memory.add_message("user", f"第二轮问题{i}" * 20)
            memory.add_message("assistant", f"第二轮回答{i}" * 20)

        msgs = memory.get_messages()
        # 摘要消息应包含合并内容
        summary_msg = msgs[1]
        assert "[历史对话摘要]" in summary_msg["content"]
        # 两次摘要都应存在
        assert "第1次摘要" in summary_msg["content"]
        assert "第2次摘要" in summary_msg["content"]

    @patch("memory._get_client")
    def test_compress_failure_skips(self, mock_get_client, memory):
        """摘要生成失败时跳过压缩，不丢失消息"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("LLM 不可用")
        mock_get_client.return_value = mock_client

        # 添加消息触发压缩
        for i in range(8):
            memory.add_message("user", f"问题{i}" * 20)
            memory.add_message("assistant", f"回答{i}" * 20)

        # 消息不应丢失（压缩失败跳过）
        assert len(memory.get_messages()) > 2


# ── 边界条件测试 ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """测试边界条件"""

    def test_find_compress_boundary_insufficient_rounds(self, memory):
        """user 消息不足 2 轮时，不压缩"""
        memory.add_message("user", "只有一个问题")
        idx = memory._find_compress_boundary()
        assert idx == 1  # 返回 1 表示早期消息太少

    def test_find_compress_boundary_exact_rounds(self, memory):
        """恰好 2 轮 user 消息"""
        memory.add_message("user", "问题1")
        memory.add_message("assistant", "回答1")
        memory.add_message("user", "问题2")
        memory.add_message("assistant", "回答2")
        idx = memory._find_compress_boundary()
        # 应返回第 2 个 user 消息的位置
        assert memory.get_messages()[idx]["content"] == "问题1"

    def test_touch_updates_timestamp(self, memory):
        before = memory.last_active_at
        time.sleep(0.01)
        memory.touch()
        assert memory.last_active_at > before

    def test_add_message_updates_timestamp(self, memory):
        before = memory.last_active_at
        time.sleep(0.01)
        memory.add_message("user", "测试")
        assert memory.last_active_at > before

    def test_no_compress_when_below_threshold(self):
        """token 数未达阈值时不触发压缩"""
        mem = ShortTermMemory(
            system_prompt="助手",
            max_tokens=100000,  # 极大阈值
            compress_threshold=0.8,
        )
        for i in range(5):
            mem.add_message("user", f"问题{i}")
            mem.add_message("assistant", f"回答{i}")
        # 不应有摘要消息
        msgs = mem.get_messages()
        assert all("[历史对话摘要]" not in m.get("content", "") for m in msgs)


# ── 格式化测试 ──────────────────────────────────────────────────────────────

class TestFormatForSummary:
    """测试消息格式化逻辑"""

    def test_observation_truncated(self):
        msgs = [
            {"role": "user", "content": "Observation: " + "很长的内容" * 100},
        ]
        result = ShortTermMemory._format_for_summary(msgs)
        assert "..." in result
        assert len(result) < 300

    def test_normal_message_preserved(self):
        msgs = [
            {"role": "user", "content": "茅台毛利率是多少？"},
            {"role": "assistant", "content": "91.96%"},
        ]
        result = ShortTermMemory._format_for_summary(msgs)
        assert "茅台" in result
        assert "91.96%" in result
