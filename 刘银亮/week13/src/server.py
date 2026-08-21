"""
FastAPI 服务：提供 SSE 流式对话接口和会话/记忆/技能管理端点

接口：
  POST /chat            - 主对话接口，SSE 流式返回每步
  POST /session/create  - 创建新会话
  GET  /session/{id}    - 获取会话信息
  DELETE /session/{id}  - 删除会话
  GET  /sessions        - 列出所有会话
  GET  /memory          - 查看当前记忆内容
  POST /memory          - 手动更新记忆
  GET  /skills          - 列出已加载的技能
  GET  /skills/{name}   - 查看单个技能详情
  POST /skills/reload   - 热重载技能
  GET  /health          - 健康检查

使用方式：
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── 应用生命周期 ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预加载 skills"""
    from skill import reload as reload_skills
    result = reload_skills()
    logger.info(f"预加载 skills 完成: {result['count']} 个 - {result['names']}")
    yield


app = FastAPI(title="办公助手 Agent", lifespan=lifespan)

# 允许跨域（前端开发用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求模型 ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    max_steps: int = 10


class MemoryUpdateRequest(BaseModel):
    content: str


# ── SSE 工具 ──────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── 主对话接口 ────────────────────────────────────────────────────────────────
async def _stream_chat(question: str, max_steps: int, session_id: Optional[str] = None):
    """
    SSE 流式生成器：在独立线程中运行 agent.chat()，通过 asyncio.Queue 传递结果
    """
    from agent import Agent

    agent = Agent()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in agent.chat(question, session_id=session_id, max_steps=max_steps):
                queue.put_nowait(step_data)
        except Exception as e:
            queue.put_nowait({"type": "error", "error": str(e)})
        finally:
            queue.put_nowait(_SENTINEL)

    # 启动工作线程
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    yield _sse({"type": "stream_end"})


@app.post("/chat")
async def chat(req: ChatRequest):
    """主对话接口，SSE 流式返回每步"""
    return StreamingResponse(
        _stream_chat(req.question, req.max_steps, req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 会话管理接口 ───────────────────────────────────────────────────────────────
@app.post("/session/create")
async def create_session():
    from session_manager import session_manager
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    from session_manager import session_manager
    info = session_manager.get_session_info(session_id)
    if info:
        return info
    return JSONResponse({"error": "会话不存在"}, status_code=404)


@app.get("/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取会话的完整消息历史"""
    from session_manager import session_manager
    if not session_manager.session_exists(session_id):
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    messages = session_manager.get_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    from session_manager import session_manager
    success = session_manager.delete_session(session_id)
    return {"success": success, "session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    from session_manager import session_manager
    return {"sessions": session_manager.list_sessions()}


# ── 记忆管理接口 ───────────────────────────────────────────────────────────────
@app.get("/memory")
async def get_memory():
    from memory import memory_manager
    return {
        "content": memory_manager.load_memory(),
        "info": memory_manager.get_memory_info(),
    }


@app.post("/memory")
async def update_memory(req: MemoryUpdateRequest):
    from memory import memory_manager
    memory_manager.save_memory(req.content)
    return {"success": True, "info": memory_manager.get_memory_info()}


# ── 技能管理接口 ───────────────────────────────────────────────────────────────
@app.get("/skills")
async def list_skills():
    from skill import list_skills_detail
    return {"skills": list_skills_detail()}


@app.get("/skills/{name}")
async def get_skill_detail(name: str):
    from skill import load_skill_detail, get_skill_info
    info = get_skill_info(name)
    if not info:
        return JSONResponse({"error": f"技能 '{name}' 不存在"}, status_code=404)
    return {
        "name": info.name,
        "description": info.description,
        "executor": info.executor,
        "has_script": info.script_path is not None and info.script_path.exists(),
        "parameters": info.parameters,
        "usage_doc": info.usage_doc,
        "detail_text": load_skill_detail(name),
    }


@app.post("/skills/reload")
async def reload_skills():
    from skill import reload
    return reload()


# ── 健康检查 ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    from skill import get_skill_catalog
    from memory import memory_manager

    return {
        "status": "ok",
        "model": os.getenv("AGENT_MODEL", "qwen-max"),
        "skills_count": len(get_skill_catalog()),
        "memory_exists": memory_manager.get_memory_info()["exists"],
    }


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists() and HTML_PATH.stat().st_size > 0:
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>办公助手 Agent</h2><p>index.html 未配置，请使用 /docs 查看 API</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
