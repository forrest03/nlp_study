"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步（支持多轮对话）
  POST /query/fc      - Function Calling 版，流式返回每步（支持多轮对话）
  POST /session/reset - 重置指定会话
  GET  /health        - 健康检查

多轮对话：
  请求中携带 session_id，服务端维护每个 session 的对话历史。
  不传 session_id 则自动生成，响应中会返回 session_id 供后续使用。

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


# ── 会话存储（内存级，进程重启后清空）─────────────────────────────────────────
# key: session_id, value: messages list
_sessions: dict[str, list] = {}


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:   str
    max_steps:  int = 10
    session_id: str | None = None   # 多轮对话会话ID，不传则自动生成


class SessionResetRequest(BaseModel):
    session_id: str


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str, session_id: str):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    多轮对话：从 _sessions 中取出历史 messages 传入，结束后存回。
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    # 取出该会话的历史消息（首次为空）
    messages = _sessions.get(session_id)

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in react_run(question, max_steps=max_steps, messages=messages):
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({"type": "start", "question": question, "mode": mode, "session_id": session_id})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        # 终态事件（final/max_steps/error）携带完整 messages，存入会话并从响应中移除（避免重复传输）；
        # llm_call 事件的 messages 是展示用的快照，需原样透传给前端
        if step_data.get("type") in ("final", "max_steps", "error") and "messages" in step_data:
            _sessions[session_id] = step_data.pop("messages")
        yield _sse(step_data)

    yield _sse({"type": "done", "session_id": session_id})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/session/reset")
async def session_reset(req: SessionResetRequest):
    """重置指定会话的对话历史"""
    if req.session_id in _sessions:
        del _sessions[req.session_id]
        return {"status": "ok", "message": f"会话 {req.session_id} 已重置"}
    return {"status": "ok", "message": "会话不存在，无需重置"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
