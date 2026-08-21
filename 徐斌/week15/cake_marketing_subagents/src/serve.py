"""
蛋糕采集 + 营销设计 Subagent HTTP 服务（FastAPI + SSE）

启动：
  cd cake_marketing_subagents
  uvicorn src.serve:app --host 0.0.0.0 --port 8015
  浏览器开 http://localhost:8015
"""
from __future__ import annotations

import os
import sys
import json
import queue
import threading
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app):
    logger.info("蛋糕营销 subagent 服务就绪")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    from llm_client import DEFAULT_MODEL

    return {
        "status": "ok",
        "search": "browser_ddg",
        "llm": bool(os.getenv("DASHSCOPE_API_KEY")),
        "provider": "qwen",
        "model": os.getenv("AGENT_MODEL", DEFAULT_MODEL),
    }


@app.post("/query")
def query(req: QueryRequest):
    import agents

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
            push(
                {
                    "type": "subagent_done",
                    "subagent_id": sid,
                    "duration": duration,
                    "subtopic": topic,
                }
            )

        def run():
            try:
                r = agents.run_cake_research(
                    req.question,
                    on_main_step=on_main_step,
                    on_dispatch=on_dispatch,
                    on_subagent_step=on_subagent_step,
                    on_subagent_done=on_subagent_done,
                )
                push(
                    {
                        "type": "final",
                        "answer": r["final_answer"],
                        "parallel_stats": r["parallel_stats"],
                        "main_trace_len": len(r["main_trace"]),
                        "subagent_count": len(r["subagents"]),
                    }
                )
            except Exception as e:
                push(
                    {
                        "type": "error",
                        "message": f"{type(e).__name__}: {str(e)[:200]}",
                    }
                )
            finally:
                push(SENTINEL)

        threading.Thread(target=run, daemon=True).start()

        yield "data: " + json.dumps(
            {"type": "start", "question": req.question}, ensure_ascii=False
        ) + "\n\n"
        while True:
            ev = q.get()
            if ev is SENTINEL:
                yield "data: " + json.dumps(
                    {"type": "done"}, ensure_ascii=False
                ) + "\n\n"
                break
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=8015, reload=False)
