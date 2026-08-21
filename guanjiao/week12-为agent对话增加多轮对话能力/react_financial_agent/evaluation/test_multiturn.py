import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

# 当前测试只验证会话与 SSE 逻辑；开发环境未安装 Web 依赖时提供最小替身。
if importlib.util.find_spec("fastapi") is None:
    fastapi = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def _route(self, *args, **kwargs):
            return lambda function: function

        get = post = delete = _route

    class HTTPException(Exception):
        pass

    fastapi.FastAPI = FastAPI
    fastapi.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi

    responses = types.ModuleType("fastapi.responses")
    responses.HTMLResponse = type("HTMLResponse", (), {})
    responses.StreamingResponse = type("StreamingResponse", (), {})
    sys.modules["fastapi.responses"] = responses

    staticfiles = types.ModuleType("fastapi.staticfiles")
    staticfiles.StaticFiles = type("StaticFiles", (), {})
    sys.modules["fastapi.staticfiles"] = staticfiles

if importlib.util.find_spec("openai") is None:
    openai = types.ModuleType("openai")
    openai.OpenAI = type("OpenAI", (), {"__init__": lambda self, *args, **kwargs: None})
    sys.modules["openai"] = openai

import serve
import react_function_calling
import react_manual


def parse_sse(events):
    return [json.loads(event.removeprefix("data: ").strip()) for event in events]


class ConversationStoreTests(unittest.TestCase):
    def test_keeps_only_latest_configured_turns(self):
        store = serve.ConversationStore(max_sessions=2, max_turns=2)

        store.append_turn("s1", "q1", "a1")
        store.append_turn("s1", "q2", "a2")
        store.append_turn("s1", "q3", "a3")

        self.assertEqual(
            store.history("s1"),
            [
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ],
        )

    def test_evicts_least_recently_used_session(self):
        store = serve.ConversationStore(max_sessions=2, max_turns=2)
        store.append_turn("s1", "q1", "a1")
        store.append_turn("s2", "q2", "a2")
        store.history("s1")
        store.append_turn("s3", "q3", "a3")

        self.assertEqual(store.history("s2"), [])
        self.assertTrue(store.history("s1"))


class AgentHistoryTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
        ]
        self.original_tools_module = sys.modules.get("tools")
        tools_module = types.ModuleType("tools")
        tools_module.TOOLS_MAP = {}
        tools_module.TOOLS_SCHEMA = []
        sys.modules["tools"] = tools_module

    def tearDown(self):
        if self.original_tools_module is None:
            sys.modules.pop("tools", None)
        else:
            sys.modules["tools"] = self.original_tools_module

    def test_manual_agent_injects_history_before_current_question(self):
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(
                    content="Thought: 信息充分\nFinal Answer: 第二答"
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        original_client = react_manual.client
        react_manual.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=Completions())
        )
        try:
            list(react_manual.run("第二问", max_steps=1, history=self.history))
        finally:
            react_manual.client = original_client

        self.assertEqual(
            captured["messages"][1:],
            self.history + [{"role": "user", "content": "第二问"}],
        )

    def test_function_calling_agent_injects_history_before_current_question(self):
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(content="第二答", tool_calls=[])
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(message=message, finish_reason="stop")
                    ]
                )

        original_client = react_function_calling.client
        react_function_calling.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=Completions())
        )
        try:
            list(
                react_function_calling.run(
                    "第二问",
                    max_steps=1,
                    history=self.history,
                )
            )
        finally:
            react_function_calling.client = original_client

        self.assertEqual(
            captured["messages"][1:],
            self.history + [{"role": "user", "content": "第二问"}],
        )


class StreamConversationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_store = serve.conversation_store
        serve.conversation_store = serve.ConversationStore(max_sessions=10, max_turns=5)
        self.received_histories = []

        fake_module = types.ModuleType("react_manual")

        def fake_run(question, max_steps=10, history=None):
            self.received_histories.append(history)
            yield {
                "step": 1,
                "type": "final",
                "thought": "",
                "answer": f"回答：{question}",
            }

        fake_module.run = fake_run
        self.original_manual_module = sys.modules.get("react_manual")
        sys.modules["react_manual"] = fake_module

    async def asyncTearDown(self):
        serve.conversation_store = self.original_store
        if self.original_manual_module is None:
            sys.modules.pop("react_manual", None)
        else:
            sys.modules["react_manual"] = self.original_manual_module

    async def collect(self, question):
        events = []
        async for event in serve._stream_react(question, 3, "manual", "session-1"):
            events.append(event)
        return parse_sse(events)

    async def test_second_turn_receives_first_turn_history(self):
        first = await self.collect("茅台的毛利率是多少？")
        second = await self.collect("那五粮液呢？")

        self.assertEqual(first[0]["history_turns"], 0)
        self.assertEqual(second[0]["history_turns"], 1)
        self.assertEqual(
            self.received_histories[1],
            [
                {"role": "user", "content": "茅台的毛利率是多少？"},
                {"role": "assistant", "content": "回答：茅台的毛利率是多少？"},
            ],
        )
        self.assertEqual(second[-1]["turn_count"], 2)

    async def test_failed_turn_is_not_saved(self):
        fake_module = sys.modules["react_manual"]

        def failing_run(question, max_steps=10, history=None):
            yield {"type": "error", "observation": "失败"}

        fake_module.run = failing_run
        events = await self.collect("失败的问题")

        self.assertEqual(events[-1]["turn_count"], 0)
        self.assertEqual(serve.conversation_store.history("session-1"), [])


if __name__ == "__main__":
    unittest.main()
