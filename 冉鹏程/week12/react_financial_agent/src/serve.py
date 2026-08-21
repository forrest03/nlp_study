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
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 会话过期配置 ────────────────────────────────────────────────────────────
_SESSION_EXPIRE_SECONDS = 1800  # 30 分钟无活动自动清除
_CLEANUP_INTERVAL_SECONDS = 300  # 每 5 分钟清理一次

# ── 会话池：按 mode 隔离，因为两种模式的 system prompt 不同 ──────────────────
_sessions: dict = {"manual": {}, "fc": {}}


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")

    # 启动会话过期清理后台任务
    cleanup_task = asyncio.create_task(_cleanup_expired_sessions())

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str
    max_steps: int = 10
    session_id: Optional[str] = None


# ── 会话管理 ────────────────────────────────────────────────────────────────

def _get_or_create_memory(session_id: str, mode: str) -> "ShortTermMemory":
    """获取或创建指定会话的短期记忆实例"""
    from memory import ShortTermMemory
    from react_manual import SYSTEM_PROMPT
    from react_function_calling import FC_SYSTEM_PROMPT

    pool = _sessions[mode]
    if session_id not in pool:
        prompt = SYSTEM_PROMPT if mode == "manual" else FC_SYSTEM_PROMPT
        pool[session_id] = ShortTermMemory(prompt)
        logger.info("创建新会话: session_id=%s, mode=%s", session_id, mode)
    return pool[session_id]


async def _cleanup_expired_sessions() -> None:
    """后台任务：定期清理超过 30 分钟无活动的会话"""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        now = time.time()
        for mode in list(_sessions.keys()):
            expired = [
                sid for sid, mem in _sessions[mode].items()
                if now - mem.last_active_at > _SESSION_EXPIRE_SECONDS
            ]
            for sid in expired:
                del _sessions[mode][sid]
                logger.info("会话过期清理: session_id=%s, mode=%s", sid, mode)


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str, memory: "ShortTermMemory"):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in react_run(question, max_steps=max_steps, memory=memory):
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({"type": "start", "question": question, "mode": mode})

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
    session_id = req.session_id or "default"
    memory = _get_or_create_memory(session_id, "manual")
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", memory),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    session_id = req.session_id or "default"
    memory = _get_or_create_memory(session_id, "fc")
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", memory),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清除指定会话的短期记忆"""
    cleared = False
    for mode in _sessions:
        if session_id in _sessions[mode]:
            del _sessions[mode][session_id]
            cleared = True
            logger.info("手动清除会话: session_id=%s, mode=%s", session_id, mode)
    return {"session_id": session_id, "cleared": cleared}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
