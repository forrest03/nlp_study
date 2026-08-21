import json
import re
import uuid
import time
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

# Allow sibling module imports
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import models
import skill_loader
import llm_client

_sessions: dict[str, dict] = {}

MAX_TOOL_ROUNDS = 10

# ──────────────────────────────────────────────
# 工具注册表: tool_name -> {skill, handler}
# ──────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, dict] = {}


def _register_tools():
    """根据已注册的 Skills 构建统一工具注册表。"""
    global _TOOL_REGISTRY
    _TOOL_REGISTRY = {}

    # --- stock-dashboard ---
    _TOOL_REGISTRY["fetch_stock"] = {
        "skill": "stock-dashboard",
        "handler": _tool_fetch_stock,
    }

    # --- flash-card ---
    _TOOL_REGISTRY["save_flashcard_data"] = {
        "skill": "flash-card",
        "handler": _tool_save_flashcard_data,
    }
    _TOOL_REGISTRY["generate_flashcard"] = {
        "skill": "flash-card",
        "handler": _tool_generate_flashcard,
    }

    # --- built-in ---
    _TOOL_REGISTRY["list_all_skills"] = {
        "skill": "__builtin__",
        "handler": _tool_list_all_skills,
    }
    _TOOL_REGISTRY["list_skill_files"] = {
        "skill": "__builtin__",
        "handler": _tool_list_skill_files,
    }


# ──────────────────────────────────────────────
# 工具实现
# ──────────────────────────────────────────────

def _tool_list_all_skills(arguments: dict) -> dict:
    skills = skill_loader.list_skills()
    return {
        "result": [
            {"name": s.name, "description": s.description, "steps": [st.title for st in s.steps]}
            for s in skills
        ]
    }


def _tool_list_skill_files(arguments: dict) -> dict:
    skill_name = arguments.get("skill_name", "")
    if not skill_name:
        return {"error": "缺少 skill_name 参数"}
    skill_path = skill_loader.get_skill_path(skill_name)
    if not skill_path:
        return {"error": f"Skill 路径不存在: {skill_name}"}
    structure = _scan_dir(skill_path)
    return {"skill": skill_name, "files": structure}


def _path_to_url(skill_name: str, abs_path: str) -> str:
    """将绝对路径转换为可访问的 URL 路径。"""
    skill_path = skill_loader.get_skill_path(skill_name)
    if not skill_path:
        return abs_path
    try:
        rel = os.path.relpath(abs_path, skill_path)
        return f"/files/skills/{skill_name}/{rel.replace(os.sep, '/')}"
    except ValueError:
        return abs_path


def _extract_urls_from_output(skill_name: str, output: str) -> list[str]:
    """从工具输出中提取文件路径并转换为 URL。"""
    urls = []
    seen = set()
    skill_path = skill_loader.get_skill_path(skill_name)
    if not skill_path:
        return urls

    # Match patterns like:
    #   [save_json] /abs/path/to/file.json
    #   [save_html] /abs/path/to/file.html
    #   [from_json] 读取已有数据：/abs/path
    for m in re.finditer(r'\[(save_html|save_json|生成|from_json)\]\s*(?:.*?[：:]\s*)?([^\s，,，。\n]+)', output):
        path_part = m.group(2).strip()
        if path_part and os.path.exists(path_part):
            url = _path_to_url(skill_name, path_part)
            if url not in seen:
                seen.add(url)
                urls.append(url)

    # Also match Chinese prefix patterns
    for line in output.split("\n"):
        line = line.strip()
        for prefix in ["已生成", "生成文件", "输出文件"]:
            if prefix in line:
                rest = line.split(prefix, 1)[-1].strip()
                if "：" in rest:
                    path_part = rest.split("：")[-1].strip()
                elif ":" in rest:
                    path_part = rest.split(":")[-1].strip()
                else:
                    path_part = rest
                if path_part and os.path.exists(path_part):
                    url = _path_to_url(skill_name, path_part)
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
                elif path_part:
                    candidate = os.path.join(skill_path, path_part)
                    if os.path.exists(candidate):
                        url = _path_to_url(skill_name, candidate)
                        if url not in seen:
                            seen.add(url)
                            urls.append(url)

    return urls


def _tool_fetch_stock(arguments: dict) -> dict:
    company = arguments.get("company", "")
    date = arguments.get("date", "")
    if not company or not date:
        return {"error": "缺少必要参数: company 和 date"}

    skill_path = skill_loader.get_skill_path("stock-dashboard")
    script_dir = os.path.join(skill_path, "scripts")
    cmd_parts = [sys.executable, os.path.join(script_dir, "fetch_stock.py"), "--company", company, "--date", date]

    if arguments.get("from_json"):
        cmd_parts.append("--from-json")
    if arguments.get("skip_html"):
        cmd_parts.append("--skip-html")

    try:
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True,
            cwd=script_dir, timeout=120, encoding="locale",
        )
        output = result.stdout + result.stderr
        status = "completed" if result.returncode == 0 else "failed"

        generated_urls = _extract_urls_from_output("stock-dashboard", output)

        return {
            "tool": "fetch_stock",
            "command": " ".join(cmd_parts),
            "status": status,
            "output": output[-3000:],
            "returncode": result.returncode,
            "generated_urls": generated_urls,
        }
    except subprocess.TimeoutExpired:
        return {"tool": "fetch_stock", "status": "timeout", "error": "执行超时(120s)"}
    except Exception as e:
        return {"tool": "fetch_stock", "status": "error", "error": str(e)}


def _tool_save_flashcard_data(arguments: dict) -> dict:
    """保存单词闪卡 JSON 数据到 flash-card/json_data/ 目录。

    说明：本工具仅负责持久化数据。调用方（LLM）需利用自身语言知识
    为单词填写完整的音标、词性、释义、3 条中英对照例句、近义词。
    """
    word = arguments.get("word", "")
    if not word:
        return {"error": "缺少 word 参数"}

    skill_path = skill_loader.get_skill_path("flash-card")
    data_dir = os.path.join(skill_path, "json_data")
    os.makedirs(data_dir, exist_ok=True)

    data = {
        "word": word,
        "phonetic": arguments.get("phonetic", ""),
        "pos": arguments.get("pos", ""),
        "definition": arguments.get("definition", ""),
        "examples": arguments.get("examples", []),
        "synonyms": arguments.get("synonyms", []),
    }

    file_path = os.path.join(data_dir, f"{word.lower()}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[save_json] {file_path}")

    file_url = _path_to_url("flash-card", file_path)

    return {
        "tool": "save_flashcard_data",
        "status": "completed",
        "file": file_path,
        "file_url": file_url,
        "generated_urls": [file_url],
        "word": word,
    }


def _tool_generate_flashcard(arguments: dict) -> dict:
    """运行 make_flashcard.py 从 JSON 生成 HTML 闪卡。"""
    word = arguments.get("word", "")
    if not word:
        return {"error": "缺少 word 参数"}

    skill_path = skill_loader.get_skill_path("flash-card")
    data_file = os.path.join(skill_path, "json_data", f"{word.lower()}.json")

    if not os.path.exists(data_file):
        return {"error": f"数据文件不存在: {data_file}，请先调用 save_flashcard_data"}

    html_dir = os.path.join(skill_path, "html_data")
    os.makedirs(html_dir, exist_ok=True)
    html_output = os.path.join(html_dir, f"{word.lower()}.html")

    scripts_dir = os.path.join(skill_path, "scripts")
    cmd_parts = [sys.executable, os.path.join(scripts_dir, "make_flashcard.py"), data_file, "-o", html_output]

    try:
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True,
            cwd=scripts_dir, timeout=30, encoding="locale",
        )
        output = result.stdout + result.stderr
        status = "completed" if result.returncode == 0 else "failed"

        generated_url = _path_to_url("flash-card", html_output) if os.path.exists(html_output) else ""

        return {
            "tool": "generate_flashcard",
            "status": status,
            "output": output[-2000:],
            "returncode": result.returncode,
            "html_generated": word,
            "generated_urls": [generated_url] if generated_url else [],
        }
    except subprocess.TimeoutExpired:
        return {"tool": "generate_flashcard", "status": "timeout", "error": "执行超时(30s)"}
    except Exception as e:
        return {"tool": "generate_flashcard", "status": "error", "error": str(e)}


# ──────────────────────────────────────────────
# 工具定义 (给 LLM 的 tools schema)
# ──────────────────────────────────────────────

def _build_all_tools() -> list[dict]:
    """构建所有 Skills 的工具定义。"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_stock",
                "description": "获取股票30分钟K线数据并生成HTML看板页面。执行后会在json_data/保存JSON、html_data/保存HTML。适用于stock-dashboard Skill。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string",
                            "description": "A股公司名称，如 平安银行、贵州茅台",
                        },
                        "date": {
                            "type": "string",
                            "description": "交易日期，格式 YYYY-MM-DD，如 2026-07-28",
                        },
                        "from_json": {
                            "type": "boolean",
                            "description": "是否从已有JSON重新生成HTML（不联网），默认false",
                        },
                        "skip_html": {
                            "type": "boolean",
                            "description": "仅获取JSON跳过HTML生成，默认false",
                        },
                    },
                    "required": ["company", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_flashcard_data",
                "description": (
                    "保存英语单词闪卡的 JSON 数据。调用此工具时，你必须利用自身语言知识"
                    "为该单词填写完整的音标、词性、中文释义、恰好3条中英对照例句、4-6个近义词。"
                    "例句需地道、能体现典型用法。适用于 flash-card Skill。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "英语单词（小写），如 crazy、resilient",
                        },
                        "phonetic": {
                            "type": "string",
                            "description": "音标，如 /rɪˈzɪliənt/",
                        },
                        "pos": {
                            "type": "string",
                            "description": "词性，如 adj., n., v.",
                        },
                        "definition": {
                            "type": "string",
                            "description": "中文释义",
                        },
                        "examples": {
                            "type": "array",
                            "description": "恰好3条中英对照例句，每条含 en 和 zh",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "en": {"type": "string", "description": "英文例句"},
                                    "zh": {"type": "string", "description": "中文翻译"},
                                },
                            },
                        },
                        "synonyms": {
                            "type": "array",
                            "description": "近义词列表（4-6个）",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["word", "phonetic", "pos", "definition", "examples", "synonyms"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_flashcard",
                "description": (
                    "根据已保存的 JSON 数据生成 HTML 闪卡页面并输出到 html_data/ 目录。"
                    "适用于 flash-card Skill。必须在 save_flashcard_data 之后调用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word": {
                            "type": "string",
                            "description": "英语单词（小写），需已通过 save_flashcard_data 保存过数据",
                        },
                    },
                    "required": ["word"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_all_skills",
                "description": "列出系统中所有可用的 Skill 及其描述",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_skill_files",
                "description": "列出指定 Skill 目录下的文件结构",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill 名称，如 stock-dashboard、flash-card",
                        },
                    },
                    "required": ["skill_name"],
                },
            },
        },
    ]
    return tools


# ──────────────────────────────────────────────
# 工具调度
# ──────────────────────────────────────────────

def _execute_tool(tool_name: str, arguments: dict) -> dict:
    """根据工具名查找 handler 并执行。"""
    _register_tools()
    entry = _TOOL_REGISTRY.get(tool_name)
    if not entry:
        return {"error": f"未知工具: {tool_name}"}
    try:
        return entry["handler"](arguments)
    except Exception as e:
        return {"error": f"工具执行异常: {e}"}


def _scan_dir(path: str, depth: int = 0, max_depth: int = 3) -> list:
    items = []
    if depth >= max_depth:
        return items
    try:
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            is_dir = os.path.isdir(full)
            item = {"name": entry, "type": "dir" if is_dir else "file"}
            if is_dir:
                item["children"] = _scan_dir(full, depth + 1, max_depth)
            items.append(item)
    except PermissionError:
        pass
    return items


# ──────────────────────────────────────────────
# SkillExecutor
# ──────────────────────────────────────────────

class SkillExecutor:
    def __init__(self):
        pass

    def create_session(self, skill_name: str, user_input: str = "", parameters: Optional[dict] = None) -> dict:
        skill_info = skill_loader.get_skill(skill_name)
        if not skill_info:
            raise ValueError(f"Skill 不存在: {skill_name}")

        session_id = str(uuid.uuid4())
        total_steps = len(skill_info.steps)
        session = {
            "session_id": session_id,
            "skill_name": skill_name,
            "skill_info": skill_info.model_dump(),
            "current_step": 0,
            "total_steps": total_steps,
            "status": "pending",
            "parameters": parameters or {},
            "user_input": user_input,
            "messages": [],
            "tool_calls_made": [],
            "results": [],
            "created_at": time.time(),
        }
        _sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[dict]:
        return _sessions.get(session_id)

    def chat(self, req: models.ChatRequest) -> models.ChatResponse:
        _register_tools()

        skills = skill_loader.list_skills()
        skills_info = [s.model_dump() for s in skills]

        system_prompt = llm_client.build_system_prompt(skills_info)

        messages = [{"role": "system", "content": system_prompt}]
        for m in req.messages:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": req.user_input})

        tools = _build_all_tools()

        try:
            final_reply, tool_records, skill_used = self._run_tool_loop(
                messages, tools
            )
        except ValueError as e:
            return models.ChatResponse(reply=str(e))
        except RuntimeError as e:
            return models.ChatResponse(reply=f"LLM 调用失败: {e}")

        skill_info = skill_loader.get_skill(skill_used) if skill_used else None

        return models.ChatResponse(
            reply=final_reply,
            next_step=None,
            action_suggestion=None,
            skill_info=skill_info,
            skill_used=skill_used,
            tool_calls=tool_records,
        )

    def _run_tool_loop(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[str, list[models.ToolCallRecord], Optional[str]]:
        tool_records: list[models.ToolCallRecord] = []
        skill_used: Optional[str] = None

        for _ in range(MAX_TOOL_ROUNDS):
            result = llm_client.chat(messages, tools=tools)

            assistant_msg = {"role": "assistant", "content": result["content"]}
            if result["tool_calls"]:
                assistant_msg["tool_calls"] = result["tool_calls"]
            messages.append(assistant_msg)

            if not result["tool_calls"]:
                return result["content"] or "(模型未生成回复)", tool_records, skill_used

            for tc in result["tool_calls"]:
                func = tc["function"]
                tool_name = func["name"]
                try:
                    arguments = json.loads(func["arguments"]) if func["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}

                tool_start = time.time()

                try:
                    tool_result = _execute_tool(tool_name, arguments)
                except Exception as e:
                    tool_result = {"error": str(e)}

                tool_duration = int((time.time() - tool_start) * 1000)

                entry = _TOOL_REGISTRY.get(tool_name, {})
                if entry.get("skill") and entry["skill"] != "__builtin__":
                    skill_used = entry["skill"]

                summary = self._summarize_tool_result(tool_name, tool_result, tool_duration)

                tool_records.append(models.ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    status=tool_result.get("status", "completed"),
                    duration_ms=tool_duration,
                    summary=summary,
                ))

                tool_reply = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                messages.append(tool_reply)

        last_msg = messages[-1]
        return last_msg.get("content", "") or "(达到最大工具调用轮次)", tool_records, skill_used

    def _summarize_tool_result(self, tool_name: str, result: dict, duration_ms: int) -> str:
        status = result.get("status", "completed")
        urls = result.get("generated_urls", [])
        if status == "completed":
            url_info = f"，生成 {len(urls)} 个文件" if urls else ""
            return f"✅ 成功，耗时 {duration_ms}ms{url_info}"
        elif status == "failed":
            return f"❌ 失败，耗时 {duration_ms}ms"
        elif status == "timeout":
            return f"⏱ 超时，耗时 {duration_ms}ms"
        else:
            return f"⚠️ {status}，耗时 {duration_ms}ms"

    def get_current_step_info(self, skill_name: str) -> Optional[dict]:
        skill_info = skill_loader.get_skill(skill_name)
        if not skill_info or not skill_info.steps:
            return None
        return {
            "skill": skill_info.name,
            "total_steps": len(skill_info.steps),
            "steps": [s.model_dump() for s in skill_info.steps],
        }

    def get_step_detail(self, skill_name: str, step_num: int) -> Optional[dict]:
        skill_info = skill_loader.get_skill(skill_name)
        if not skill_info:
            return None
        for step in skill_info.steps:
            if step.step_num == step_num:
                return {"step": step.model_dump(), "skill_name": skill_name}
        return None


_executor = SkillExecutor()


def get_executor() -> SkillExecutor:
    return _executor
