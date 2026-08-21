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
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

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
    question:   str
    max_steps:  int = 10
    session_id: Optional[str] = None


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str, session_id: Optional[str] = None):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    新增多轮对话支持：
      - session_id 为空时创建新会话
      - session_id 存在时从历史上下文恢复
      - 每轮结束后保存对话历史
      - 应用上下文管理策略压缩历史
    """
    from session_manager import session_manager, context_manager

    # 处理会话：创建或恢复
    if not session_id or not session_manager.session_exists(session_id):
        session_id = session_manager.create_session()
        logger.info(f"创建新会话: {session_id}")
    else:
        logger.info(f"恢复会话: {session_id}")

    # 获取历史消息并应用压缩策略
    history_messages = session_manager.get_messages(session_id)
    if history_messages:
        history_messages = context_manager.apply_strategy(history_messages, "smart_compact")
        logger.debug(f"历史消息: {len(history_messages)} 条（已压缩）")

    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    final_messages: list = []  # 用于收集最终消息

    def _worker():
        nonlocal final_messages
        try:
            # 将历史消息和新消息传递给 run()
            for step_data in react_run(question, max_steps=max_steps, history_messages=history_messages):
                queue.put_nowait(step_data)
                # 收集最终消息（包含完整 messages 列表）
                if step_data.get("type") in ("final", "error", "max_steps"):
                    final_messages = step_data.get("messages", [])
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({"type": "start", "question": question, "mode": mode, "session_id": session_id})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        # 在 SSE 事件中携带 session_id
        step_data["session_id"] = session_id
        yield _sse(step_data)

    # 保存最终消息到会话
    if final_messages:
        session_manager.save_messages(session_id, final_messages)
        logger.info(f"保存会话 {session_id} 消息: {len(final_messages)} 条")
    session_manager.increment_turn(session_id)
    logger.info(f"会话 {session_id} 轮次增加，当前轮次: {session_manager.get_session_info(session_id)['turn_count']}")

    yield _sse({"type": "done", "session_id": session_id})


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


# ── 会话管理接口 ───────────────────────────────────────────────────────────────
@app.post("/session/create")
async def create_session():
    """创建新会话"""
    from session_manager import session_manager
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    from session_manager import session_manager
    success = session_manager.delete_session(session_id)
    return {"success": success, "session_id": session_id}


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    from session_manager import session_manager
    info = session_manager.get_session_info(session_id)
    if info:
        return info
    return {"error": "会话不存在"}, 404


@app.get("/sessions")
async def list_sessions():
    """列出所有活跃会话"""
    from session_manager import session_manager
    return {"sessions": session_manager.list_sessions()}


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
