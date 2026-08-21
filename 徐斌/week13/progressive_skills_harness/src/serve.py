"""
FastAPI + SSE：可视化渐进式加载过程。

启动：
  cd progressive_skills_harness
  uvicorn src.serve:app --host 0.0.0.0 --port 8013
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from harness import SkillsHarness  # noqa: E402
from llm_config import current_model_info  # noqa: E402
from skill_registry import SkillRegistry  # noqa: E402

app = FastAPI(title="Progressive Skills Harness")
registry = SkillRegistry(ROOT / "skills")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_steps: int = Field(12, ge=1, le=30)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/skills")
def api_skills():
    return {
        "model": current_model_info(),
        "index": registry.build_index_text(),
        "skills": [
            {
                "name": m.name,
                "description": m.description,
                "triggers": m.triggers,
                "path": str(m.path.relative_to(ROOT)),
            }
            for m in registry.list_metas()
        ],
    }


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    harness = SkillsHarness(max_steps=req.max_steps)

    def event_stream():
        try:
            for ev in harness.run(req.message):
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# 方便预览 workspace 产物
workspace = ROOT / "workspace"
workspace.mkdir(exist_ok=True)
app.mount("/workspace", StaticFiles(directory=str(workspace)), name="workspace")
