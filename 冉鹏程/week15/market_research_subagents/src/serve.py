"""企业信息调查 Agent 的 FastAPI + SSE 服务。

服务将主 Agent、六个并行核验子 Agent 的 ReAct 步骤和最终求职尽调报告实时推送到前端。
启动命令：uvicorn src.serve:app --host 0.0.0.0 --port 8002。
"""
import json
import logging
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from config import load_project_environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
STATIC_DIR = BASE_DIR / "static"

load_project_environment()


@asynccontextmanager
async def lifespan(app):
    logger.info("企业信息调查 Agent 服务就绪")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    """企业调查接口的输入模型，负责限制外部文本长度。"""

    question: str = Field(min_length=2, max_length=240, description="企业名称及可选求职关注点")


@app.get("/health")
def health():
    """返回服务及所需外部凭据是否就绪，不暴露任何密钥内容。"""
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    has_llm = bool(os.getenv("DEEPSEEK_API_KEY"))
    return {"status": "ok", "tavily": has_tavily, "llm": has_llm}


@app.post("/query")
def query(req: QueryRequest):
    """以 SSE 流式执行企业尽调。

    参数：req 为已完成长度校验的企业调查请求。
    返回：主从 Agent 的实时事件与最终报告。
    异常：清理后为空的请求返回 HTTP 422。
    """
    import agents

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="企业调查请求不能为空")
    request_id = uuid.uuid4().hex
    logger.info("company_query_received request_id=%s question_length=%s", request_id, len(question))

    def event_stream():
        q = queue.Queue()
        SENTINEL = object()

        def push(ev):
            q.put(ev)

        def on_main_step(step):
            push({"type": "main_step", **step})

        def on_dispatch(info):
            push({"type": "dispatch", **info})

        def on_subagent_step(sid, step):
            push({"type": "subagent_step", "subagent_id": sid, **step})

        def on_subagent_done(sid, duration, topic):
            push({"type": "subagent_done", "subagent_id": sid,
                  "duration": duration, "subtopic": topic})

        def run():
            try:
                r = agents.run_research(
                    question,
                    on_main_step=on_main_step,
                    on_dispatch=on_dispatch,
                    on_subagent_step=on_subagent_step,
                    on_subagent_done=on_subagent_done,
                    request_id=request_id,
                )
                push({"type": "final", "answer": r["final_answer"],
                      "parallel_stats": r["parallel_stats"],
                      "main_trace_len": len(r["main_trace"]),
                      "subagent_count": len(r["subagents"])})
            except Exception as e:
                logger.exception("company_query_failed request_id=%s", request_id)
                push({"type": "error", "message": f"{type(e).__name__}: {str(e)[:200]}"})
            finally:
                push(SENTINEL)

        threading.Thread(target=run, daemon=True).start()

        # 先发 start
        yield "data: " + json.dumps({"type": "start", "question": question, "request_id": request_id},
                                    ensure_ascii=False) + "\n\n"
        while True:
            ev = q.get()
            if ev is SENTINEL:
                yield "data: " + json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
                break
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8002, reload=False)
