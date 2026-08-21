"""
Harness Web 服务（端口 8000）

接口：
  GET  /                     前端页面
  GET  /health               健康检查
  POST /api/session/new      创建新会话
  GET  /api/sessions         会话列表
  GET  /api/session/{id}     会话历史
  POST /api/chat             对话（支持 SSE 流式）
  GET  /api/memory/{type}    读取记忆
  PUT  /api/memory/{type}    更新记忆
  POST /api/memory/compress  压缩长期记忆
  POST /api/memory/rebuild   重建检索索引
  POST /api/confirm-tool/{id}  确认执行工具命令
  POST /api/reject-tool/{id}   拒绝执行工具命令
  POST /api/upload           上传文件
  DELETE /api/upload/{filename}  删除已上传文件

启动：
  python harness/serve.py
  uvicorn harness.serve:app --host 0.0.0.0 --port 8000
"""

import asyncio
import imghdr
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = ROOT / "files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_READERS = {
    "user_profile": lambda s: s.read_user_profile(),
    "long_term_raw": lambda s: s.read_long_term_raw_md(),
    "compressed": lambda s: s.read_compressed_md(),
    "daily": lambda s: s.read_daily(),
    "memory_meta": lambda s: s.read_memory_meta_raw(),
}

MEMORY_WRITERS = {
    "user_profile": lambda s, c: s.update_user_profile(c),
    "long_term_raw": lambda s, c: s.write_long_term_raw_md(c),
    "compressed": lambda s, c: s.write_compressed_md(c),
    "daily": lambda s, c: s.write_daily(c),
    "memory_meta": lambda s, c: s.write_memory_meta_raw(c),
}

_pending_confirmations: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from harness.chat_harness import init_demo_vectorstore
    from src.memory_retriever import reload_retriever

    logger.info("初始化向量索引...")
    await asyncio.to_thread(init_demo_vectorstore)
    await asyncio.to_thread(reload_retriever)
    logger.info("Web 服务就绪 → http://0.0.0.0:8000")
    yield


app = FastAPI(title="Harness Memory Chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/files", StaticFiles(directory=str(UPLOAD_DIR)), name="files")


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    stream: bool = True
    allow_tools: bool = False


class MemoryUpdateRequest(BaseModel):
    content: str
    date: Optional[str] = None


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "harness-memory-chat"}


@app.get("/api/skills")
async def list_skills():
    from harness.tool_executor import get_skill_descriptions
    from skills import discover_skills

    skills = discover_skills()
    result = []
    for name, sk in skills.items():
        entry = {
            "name": sk.name,
            "description": sk.description,
            "tools": sk.tools,
        }
        result.append(entry)
    return {"skills": result}


@app.post("/api/session/new")
async def create_session():
    from src.memory_store import get_memory_store

    store = get_memory_store()
    session_id = await asyncio.to_thread(store.create_session)
    return {"session_id": session_id}


@app.get("/api/sessions")
async def list_sessions():
    from src.memory_store import get_memory_store

    store = get_memory_store()
    sessions = await asyncio.to_thread(store.list_sessions)
    return {"sessions": sessions}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    from src.memory_store import get_memory_store

    store = get_memory_store()
    turns = await asyncio.to_thread(store.get_short_term, session_id)
    return {"session_id": session_id, "turns": turns}


@app.post("/api/confirm-tool/{confirm_id}")
async def confirm_tool(confirm_id: str):
    if confirm_id not in _pending_confirmations:
        raise HTTPException(404, "确认请求不存在或已过期")
    _pending_confirmations[confirm_id]["approved"] = True
    _pending_confirmations[confirm_id]["event"].set()
    return {"ok": True}


@app.post("/api/reject-tool/{confirm_id}")
async def reject_tool(confirm_id: str):
    if confirm_id not in _pending_confirmations:
        raise HTTPException(404, "确认请求不存在或已过期")
    _pending_confirmations[confirm_id]["approved"] = False
    _pending_confirmations[confirm_id]["event"].set()
    return {"ok": True}


_ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif",
    ".ppt", ".pptx", ".doc", ".docx",
    ".wsl", ".wslx",
}

# Magic bytes: (extension_set, min_bytes, checker_func)
# checker_func receives the raw header bytes, returns True if valid
_MAGIC_CHECKERS: list[tuple[set, int, callable]] = [
    # JPEG: FF D8 FF
    ({".jpg", ".jpeg"}, 3, lambda h: h[:3] == b"\xff\xd8\xff"),
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    ({".png"}, 8, lambda h: h[:8] == b"\x89PNG\r\n\x1a\n"),
    # GIF: GIF87a / GIF89a
    ({".gif"}, 6, lambda h: h[:6] in (b"GIF87a", b"GIF89a")),
    # OLE2 Compound Document (old .ppt, .doc)
    ({".ppt", ".doc"}, 8, lambda h: h[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"),
    # ZIP-based (pptx, docx, wslx — modern Office / OOXML)
    ({".pptx", ".docx", ".wslx"}, 4, lambda h: h[:4] == b"PK\x03\x04"),
]


def _validate_file(filename: str, content: bytes) -> str | None:
    """Return error string if invalid, or None if file passes checks."""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return f"不支持的文件格式: {ext}（允许: {', '.join(sorted(_ALLOWED_EXTENSIONS))}）"

    if len(content) == 0:
        return "文件内容为空"

    # binary header check
    for exts, nbytes, check in _MAGIC_CHECKERS:
        if ext in exts:
            if len(content) < nbytes:
                return f"文件过小，无法验证格式: {filename}"
            if not check(content[:nbytes]):
                return f"文件头校验失败，实际内容与 {ext} 格式不符: {filename}"
            break
    else:
        # .wsl only — no magic known, skip binary check
        pass

    return None


def _ext_allowed_in_frontend(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ALLOWED_EXTENSIONS


@app.post("/api/upload")
async def upload_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")

    content = await file.read()

    error = _validate_file(file.filename, content)
    if error:
        raise HTTPException(400, error)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / file.filename
    # 避免同名覆盖 — 若已存在则添加序号
    stem, ext = Path(save_path).stem, Path(save_path).suffix
    counter = 1
    while save_path.exists():
        save_path = UPLOAD_DIR / f"{stem}_{counter}{ext}"
        counter += 1
    save_path.write_bytes(content)
    url = f"/files/{save_path.name}"
    logger.info(f"文件上传: {save_path} ({len(content)} bytes)")
    return {
        "success": True,
        "filename": save_path.name,
        "url": url,
        "path": str(save_path),
        "size": len(content),
    }


@app.delete("/api/upload/{filename:path}")
async def delete_upload(filename: str):
    save_path = UPLOAD_DIR / filename
    if not save_path.exists():
        raise HTTPException(404, "文件不存在")
    save_path.unlink()
    logger.info(f"文件删除: {save_path}")
    return {"success": True, "filename": filename}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")

    return StreamingResponse(
        _stream_chat_with_tools(req.question.strip(), req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _stream_chat(question: str, session_id: Optional[str]):
    from harness.chat_harness import _build_system_prompt, run_compression_if_needed
    from harness.llm_client import chat_stream, DEFAULT_MODEL
    from src.memory_retriever import reload_retriever
    from src.memory_store import get_memory_store

    store = get_memory_store()

    if not session_id:
        session_id = store.create_session()

    yield _sse({"type": "start", "session_id": session_id, "question": question})

    system_prompt = _build_system_prompt(question, session_id)
    history = store.short_term_to_messages(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    parts: list[str] = []

    def _worker():
        for chunk in chat_stream(messages, model=DEFAULT_MODEL):
            parts.append(chunk)

    task = asyncio.create_task(asyncio.to_thread(_worker))
    sent = 0
    while not task.done() or sent < len(parts):
        while sent < len(parts):
            yield _sse({"type": "token", "content": parts[sent]})
            sent += 1
        if not task.done():
            await asyncio.sleep(0.03)

    answer = "".join(parts)
    store.add_short_term_turn(session_id, question, answer)

    if run_compression_if_needed(session_id=session_id):
        reload_retriever()
        yield _sse({"type": "info", "message": "长期记忆已自动压缩"})

    yield _sse({"type": "done", "session_id": session_id, "answer": answer})
@app.get("/api/memory/{memory_type}")
async def get_memory(memory_type: str, day: Optional[str] = None):
    if memory_type not in MEMORY_READERS:
        raise HTTPException(404, f"未知记忆类型: {memory_type}")

    from src.memory_store import get_memory_store

    store = get_memory_store()
    if memory_type == "daily" and day:
        content = await asyncio.to_thread(store.read_daily, date.fromisoformat(day))
    else:
        content = await asyncio.to_thread(MEMORY_READERS[memory_type], store)
    return {"type": memory_type, "content": content}


@app.put("/api/memory/{memory_type}")
async def update_memory(memory_type: str, req: MemoryUpdateRequest):
    if memory_type not in MEMORY_WRITERS:
        raise HTTPException(404, f"未知记忆类型: {memory_type}")

    from src.memory_store import get_memory_store

    store = get_memory_store()
    if memory_type == "daily":
        d = date.fromisoformat(req.date) if req.date else date.today()
        await asyncio.to_thread(store.write_daily, req.content, d)
    elif memory_type == "memory_meta":
        try:
            json.loads(req.content)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"memory_meta 必须是合法 JSON: {e}") from e
        await asyncio.to_thread(MEMORY_WRITERS[memory_type], store, req.content)
    else:
        await asyncio.to_thread(MEMORY_WRITERS[memory_type], store, req.content)

    return {"ok": True, "type": memory_type}


@app.post("/api/memory/compress")
async def compress_memory():
    from src.memory_compressor import compress_raw_memories
    from src.memory_retriever import reload_retriever

    chunks = await asyncio.to_thread(compress_raw_memories)
    await asyncio.to_thread(reload_retriever)
    return {"ok": True, "compressed_count": len(chunks)}


@app.post("/api/memory/rebuild")
async def rebuild_index():
    from src.memory_compressor import rebuild_vector_index
    from src.memory_retriever import reload_retriever

    await asyncio.to_thread(rebuild_vector_index)
    await asyncio.to_thread(reload_retriever)
    return {"ok": True}


async def _stream_chat_with_tools(question: str, session_id: Optional[str]):
    from harness.chat_harness import _build_system_prompt, run_compression_if_needed, TOOLS_SYSTEM_PROMPT
    from harness.llm_client import chat_with_tools, DEFAULT_MODEL
    from harness.tool_executor import get_all_tools, get_skill_descriptions, ToolExecutor
    from src.memory_retriever import reload_retriever
    from src.memory_store import get_memory_store

    store = get_memory_store()

    if not session_id:
        session_id = store.create_session()

    executor = ToolExecutor(
        project_root=str(ROOT),
        allow_tools=True,
        confirm_commands=False,
    )

    yield _sse({"type": "start", "session_id": session_id, "question": question, "mode": "tools"})

    system_prompt = _build_system_prompt(question, session_id)
    skill_prompts = get_skill_descriptions()
    system_prompt += TOOLS_SYSTEM_PROMPT.format(skill_prompts=skill_prompts)
    history = store.short_term_to_messages(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    tools = get_all_tools()
    max_turns = 10
    for turn in range(max_turns):
        content, tool_calls = await asyncio.to_thread(
            chat_with_tools, messages, tools, DEFAULT_MODEL, 0.7
        )

        if tool_calls:
            tool_info = []
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_info.append(f"{name}({json.dumps(args, ensure_ascii=False)})")
            yield _sse({"type": "tool_call", "tools": tool_info, "turn": turn + 1})
            logger.info(f"[工具调用] {', '.join(tool_info)}")

        messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        } if tool_calls else {"role": "assistant", "content": content or ""})

        if not tool_calls:
            answer = content or ""
            yield _sse({"type": "token", "content": answer})
            break

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # 对 execute_command 命令要求用户手动确认
            if name == "execute_command":
                command = args.get("command", "")
                confirm_id = str(uuid.uuid4())[:8]
                event = asyncio.Event()
                _pending_confirmations[confirm_id] = {
                    "event": event, "approved": None, "command": command,
                }
                yield _sse({
                    "type": "confirmation_required",
                    "confirm_id": confirm_id,
                    "command": command,
                })
                try:
                    await asyncio.wait_for(event.wait(), timeout=120)
                except asyncio.TimeoutError:
                    _pending_confirmations.pop(confirm_id, None)
                    result = {"success": False, "error": "确认超时", "stdout": "", "stderr": ""}
                    result_str = json.dumps(result, ensure_ascii=False)
                    yield _sse({"type": "tool_result", "tool": name, "success": False})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                    continue
                approved = _pending_confirmations.pop(confirm_id, {}).get("approved", False)
                if not approved:
                    result = {"success": False, "error": "用户拒绝执行该命令", "stdout": "", "stderr": ""}
                    result_str = json.dumps(result, ensure_ascii=False)
                    yield _sse({"type": "tool_result", "tool": name, "success": False})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                    continue

            result = await asyncio.to_thread(executor.execute_tool, name, args)
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 10000:
                result_str = result_str[:10000] + "\n...(截断)"
            yield _sse({"type": "tool_result", "tool": name, "success": result.get("success", False)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
    else:
        answer = "工具调用次数超过上限，请简化你的请求。"
        yield _sse({"type": "token", "content": answer})

    store.add_short_term_turn(session_id, question, answer)
    yield _sse({"type": "done", "session_id": session_id, "answer": answer})

    if run_compression_if_needed(session_id=session_id):
        reload_retriever()
        yield _sse({"type": "info", "message": "长期记忆已自动压缩"})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("harness.serve:app", host="0.0.0.0", port=8000, reload=False)
