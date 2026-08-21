"""
FastAPI HTTP 服务（支持会话记忆）

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步（支持会话记忆）
  POST /query/fc      - Function Calling 版，流式返回每步（支持会话记忆）
  POST /session/create - 创建新会话
  GET  /sessions       - 获取所有会话列表
  GET  /session/{id}/history - 获取指定会话的历史记录
  GET  /health        - 健康检查

使用方式：
  uvicorn src_my.serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Path as PathParam
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS 和会话存储（启动时执行一次）───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引、Embedding 模型和会话存储...")
    from tools import _load_rag
    from conversation_store import get_conv_store

    await asyncio.to_thread(_load_rag)
    await asyncio.to_thread(get_conv_store()._load)

    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent（带记忆）", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:   str
    max_steps:  int = 10
    session_id: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history:    list


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str,
                        session_id: Optional[str] = None):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    新增：支持会话记忆
    """
    from conversation_store import get_conv_store

    conv_store = get_conv_store()

    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in react_run(
                question,
                max_steps=max_steps,
                session_id=session_id,
                conv_store=conv_store,
            ):
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
        yield _sse(step_data)

    yield _sse({"type": "done"})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 会话管理路由 ──────────────────────────────────────────────────────────────
@app.post("/session/create", response_model=CreateSessionResponse)
async def create_session():
    """创建新会话"""
    from conversation_store import get_conv_store
    conv_store = get_conv_store()
    session_id = conv_store.create_session()
    return {"session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    """获取所有会话列表"""
    from conversation_store import get_conv_store
    conv_store = get_conv_store()
    return conv_store.list_sessions()


@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: str = PathParam(description="会话ID"),
):
    """获取指定会话的历史记录"""
    from conversation_store import get_conv_store
    conv_store = get_conv_store()
    history = conv_store.get_session_history(session_id)
    return {"session_id": session_id, "history": history}


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")