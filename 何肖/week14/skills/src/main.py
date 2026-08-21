import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

# Allow both direct execution and package import
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import models
import skill_loader
import skill_executor
import llm_client

BASE_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = BASE_DIR / "skills"

app = FastAPI(
    title="Harness · Skill Agent",
    description="基于 FastAPI + DashScope 的智能 Skill Agent，自动识别用户意图并调用工具",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = skill_executor.get_executor()


@app.get("/api/skills", response_model=models.SkillListResponse)
def api_list_skills():
    skills = skill_loader.list_skills()
    return models.SkillListResponse(skills=skills)


@app.get("/api/skills/{skill_name}", response_model=models.SkillDetailResponse)
def api_skill_detail(skill_name: str):
    skill = skill_loader.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {skill_name}")
    return models.SkillDetailResponse(skill=skill)


@app.get("/api/skills/{skill_name}/md")
def api_skill_md(skill_name: str):
    md = skill_loader.read_skill_md(skill_name)
    if not md:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {skill_name}")
    return {"skill": skill_name, "content": md}


@app.get("/api/skills/{skill_name}/steps")
def api_skill_steps(skill_name: str):
    info = executor.get_current_step_info(skill_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {skill_name}")
    return info


@app.get("/api/skills/{skill_name}/steps/{step_num}")
def api_skill_step_detail(skill_name: str, step_num: int):
    detail = executor.get_step_detail(skill_name, step_num)
    if not detail:
        raise HTTPException(status_code=404, detail=f"步骤不存在: Step {step_num}")
    return detail


@app.post("/api/chat", response_model=models.ChatResponse)
def api_chat(req: models.ChatRequest):
    """智能对话接口 - 自动识别用户意图并调用对应 Skill 的工具。"""
    return executor.chat(req)


@app.post("/api/skills/{skill_name}/run")
def api_run_skill_step(skill_name: str, req: models.ExecuteRequest):
    skill = skill_loader.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {skill_name}")

    if skill_name in ("stock-dashboard", "stock-dashboard-optimized"):
        tool_name = "fetch_stock" if skill_name == "stock-dashboard" else "fetch_stock_optimized"
        arguments = {
            "company": req.parameters.get("company", ""),
            "date": req.parameters.get("date", ""),
            "from_json": req.parameters.get("from_json", False),
            "skip_html": req.parameters.get("skip_html", False),
        }
        result = skill_executor._execute_tool(tool_name, arguments)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    raise HTTPException(status_code=400, detail=f"Skill「{skill_name}」暂不支持直接执行")


@app.get("/api/skills/{skill_name}/files")
def api_list_skill_files(skill_name: str):
    skill_path = skill_loader.get_skill_path(skill_name)
    if not skill_path:
        raise HTTPException(status_code=404, detail=f"Skill 不存在: {skill_name}")

    structure = _scan_skill_dir(skill_path)
    return {"skill": skill_name, "structure": structure}


@app.get("/files/skills/{skill_name}/{file_path:path}")
def serve_skill_file(skill_name: str, file_path: str):
    """Serve static files from skills directory (HTML dashboards, flash cards, etc.)."""
    skill_dir = SKILLS_DIR / skill_name
    if not skill_dir.exists():
        raise HTTPException(status_code=404, detail=f"Skill 目录不存在: {skill_name}")

    full_path = skill_dir / file_path
    full_path = full_path.resolve()
    skill_dir_resolved = skill_dir.resolve()

    if not str(full_path).startswith(str(skill_dir_resolved)):
        raise HTTPException(status_code=403, detail="禁止访问")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    response = FileResponse(str(full_path))
    # Disable caching for HTML files to ensure latest data is shown
    if str(full_path).endswith('.html'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


@app.get("/api/health")
def api_health():
    skills = skill_loader.list_skills()
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "dashscope_configured": bool(llm_client.DASHSCOPE_API_KEY),
        "model": llm_client.AGENT_MODEL,
        "skills_count": len(skills),
        "skills": [s.name for s in skills],
        "skills_dir": str(SKILLS_DIR),
    }


@app.get("/chat")
def chat_page():
    chat_html = Path(__file__).parent / "chat.html"
    return FileResponse(str(chat_html))


@app.get("/")
def root():
    skills = skill_loader.list_skills()
    skill_list_html = ""
    for s in skills:
        steps_html = "".join(
            f"<li>Step {st.step_num}: {st.title}</li>"
            for st in s.steps
        ) or "<li>暂无步骤</li>"
        skill_list_html += f"""
        <div class="skill-card">
            <div class="skill-name">{s.name}</div>
            <div class="skill-desc">{s.description}</div>
            <div class="skill-steps"><b>步骤：</b><ul>{steps_html}</ul></div>
            <div class="skill-path">路径：{s.path}</div>
        </div>"""

    if not skills:
        skill_list_html = '<p class="empty">暂无可用 Skill。</p>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Harness · Skill Agent</title>
        <style>
            * {{ margin:0; padding:0; box-sizing:border-box; }}
            body {{ font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                   background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                   color: #e0e0e0; min-height: 100vh; padding: 40px 20px; }}
            .container {{ max-width: 960px; margin: 0 auto; }}
            h1 {{ font-size: 28px; color: #fff; text-align: center; margin-bottom: 8px;
                  text-shadow: 0 0 20px rgba(0, 229, 255, 0.5); }}
            .subtitle {{ text-align: center; color: #00e5ff; margin-bottom: 32px; font-size: 14px; }}
            .hud {{ display:flex; justify-content:center; gap:16px; margin-bottom:24px; }}
            .hud-item {{ background: rgba(0,229,255,0.1); border: 1px solid rgba(0,229,255,0.3);
                        padding: 8px 16px; border-radius: 4px; font-size: 12px; }}
            .hud-item b {{ color: #00e5ff; }}
            .skill-card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                          border-radius: 8px; padding: 20px; margin-bottom: 16px;
                          transition: all 0.3s ease; }}
            .skill-card:hover {{ border-color: rgba(0,229,255,0.5);
                                 box-shadow: 0 0 20px rgba(0,229,255,0.2);
                                 transform: translateY(-2px); }}
            .skill-name {{ font-size: 20px; color: #00e5ff; margin-bottom: 8px; }}
            .skill-desc {{ color: #b0b0b0; margin-bottom: 12px; font-size: 14px; }}
            .skill-steps {{ color: #a0a0a0; font-size: 13px; margin-bottom: 12px; }}
            .skill-steps ul {{ margin: 8px 0 8px 20px; }}
            .skill-steps li {{ margin-bottom: 2px; }}
            .skill-path {{ color: #666; font-size: 12px; font-family: monospace; }}
            .empty {{ text-align: center; color: #888; padding: 40px; }}
            .cta {{ text-align: center; margin: 24px 0; }}
            .cta a {{ display:inline-block; background:#00e5ff; color:#000; padding:12px 32px;
                     border-radius:8px; text-decoration:none; font-weight:bold; transition:all 0.2s; }}
            .cta a:hover {{ background:#00b8d4; transform:scale(1.05); }}
            .api-section {{ margin-top: 40px; padding-top: 24px; border-top: 1px solid rgba(255,255,255,0.1); }}
            .api-section h2 {{ font-size: 18px; color: #00e5ff; margin-bottom: 16px; }}
            .api-section code {{ background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px;
                                 color: #ffcc00; font-size: 13px; display: block; margin: 4px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>◆ Harness · Skill Agent</h1>
            <p class="subtitle">FASTAPI + DASHSCOPE · 自动意图识别 · 工具调用</p>
            <div class="hud">
                <div class="hud-item">可用 Skills: <b>{len(skills)}</b></div>
                <div class="hud-item">DashScope: <b>{"✔ 已配置" if llm_client.DASHSCOPE_API_KEY else "✘ 未配置"}</b></div>
                <div class="hud-item">时间: <b>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</b></div>
            </div>
            {skill_list_html}
            <div class="cta">
                <a href="/chat">进入 Agent 对话 →</a>
            </div>
            <div class="api-section">
                <h2>API 端点</h2>
                <code>GET  /api/skills              — 列出所有 Skill</code>
                <code>POST /api/chat                — 智能对话（自动识别意图）</code>
                <code>POST /api/skills/{{name}}/run   — 直接运行 Skill 脚本</code>
                <code>GET  /api/health               — 健康检查</code>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


def _scan_skill_dir(path: str, depth: int = 0, max_depth: int = 3) -> list[dict]:
    items = []
    if depth >= max_depth:
        return items
    try:
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            is_dir = os.path.isdir(full)
            item = {
                "name": entry,
                "type": "dir" if is_dir else "file",
                "path": os.path.relpath(full, path),
            }
            if is_dir:
                item["children"] = _scan_skill_dir(full, depth + 1, max_depth)
            items.append(item)
    except PermissionError:
        pass
    return items
