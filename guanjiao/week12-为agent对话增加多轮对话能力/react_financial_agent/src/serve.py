"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  GET  /health        - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
import uuid
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    max_steps: int = Field(default=10, ge=1, le=30)
    session_id: Optional[str] = None


class ConversationStore:
    """进程内会话存储；仅保存用户消息与最终回答。"""

    def __init__(self, max_sessions: int = 200, max_turns: int = 10):
        self.max_sessions = max_sessions
        self.max_messages = max_turns * 2
        self._histories: OrderedDict[str, list[dict]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def normalize_id(self, session_id: Optional[str]) -> str:
        if session_id is None:
            return uuid.uuid4().hex
        session_id = session_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", session_id):
            raise ValueError("session_id 只能包含字母、数字、点、下划线和连字符，最长 128 位")
        return session_id

    def lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def history(self, session_id: str) -> list[dict]:
        history = self._histories.get(session_id, [])
        if session_id in self._histories:
            self._histories.move_to_end(session_id)
        return [message.copy() for message in history]

    def append_turn(self, session_id: str, question: str, answer: str) -> int:
        history = self._histories.setdefault(session_id, [])
        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        del history[:-self.max_messages]
        self._histories.move_to_end(session_id)
        self._evict()
        return len(history) // 2

    def clear(self, session_id: str) -> bool:
        return self._histories.pop(session_id, None) is not None

    def _evict(self):
        while len(self._histories) > self.max_sessions:
            oldest_id, _ = self._histories.popitem(last=False)
            lock = self._locks.get(oldest_id)
            if lock is not None and not lock.locked():
                self._locks.pop(oldest_id, None)


conversation_store = ConversationStore(
    max_sessions=int(os.getenv("MAX_CONVERSATION_SESSIONS", "200")),
    max_turns=int(os.getenv("MAX_CONVERSATION_TURNS", "10")),
)


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(
    question: str,
    max_steps: int,
    mode: str,
    session_id: Optional[str] = None,
):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。
    """
    try:
        normalized_session_id = conversation_store.normalize_id(session_id)
    except ValueError as exc:
        yield _sse({"type": "error", "observation": str(exc)})
        yield _sse({"type": "done"})
        return

    async with conversation_store.lock(normalized_session_id):
        if mode == "manual":
            from react_manual import run as react_run
        else:
            from react_function_calling import run as react_run

        history = conversation_store.history(normalized_session_id)
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def _enqueue(value):
            loop.call_soon_threadsafe(queue.put_nowait, value)

        def _worker():
            try:
                for step_data in react_run(
                    question,
                    max_steps=max_steps,
                    history=history,
                ):
                    _enqueue(step_data)
            except Exception as exc:
                logger.exception("Agent 执行失败")
                _enqueue({
                    "type": "error",
                    "observation": f"Agent 执行失败: {exc}",
                })
            finally:
                _enqueue(sentinel)

        yield _sse({
            "type": "start",
            "question": question,
            "mode": mode,
            "session_id": normalized_session_id,
            "history_turns": len(history) // 2,
        })

        loop.run_in_executor(None, _worker)
        final_answer = None

        while True:
            step_data = await queue.get()
            if step_data is sentinel:
                break
            if step_data.get("type") == "final":
                final_answer = step_data.get("answer")
            yield _sse(step_data)

        if final_answer:
            turn_count = conversation_store.append_turn(
                normalized_session_id,
                question,
                final_answer,
            )
        else:
            turn_count = len(history) // 2

        yield _sse({
            "type": "done",
            "session_id": normalized_session_id,
            "turn_count": turn_count,
        })


# ── 路由 ──────────────────────────────────────────────────────────────────────
def _question(req: QueryRequest) -> str:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")
    return question


@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(_question(req), req.max_steps, "manual", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(_question(req), req.max_steps, "fc", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    try:
        normalized_session_id = conversation_store.normalize_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    async with conversation_store.lock(normalized_session_id):
        cleared = conversation_store.clear(normalized_session_id)
    return {"session_id": normalized_session_id, "cleared": cleared}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
