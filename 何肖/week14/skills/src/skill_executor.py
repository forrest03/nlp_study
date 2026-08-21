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

    # --- stock-dashboard (原版) ---
    _TOOL_REGISTRY["fetch_stock"] = {
        "skill": "stock-dashboard",
        "handler": _tool_fetch_stock,
        "version": "original",
    }

    # --- stock-dashboard-optimized (优化版) ---
    _TOOL_REGISTRY["fetch_stock_optimized"] = {
        "skill": "stock-dashboard-optimized",
        "handler": _tool_fetch_stock_optimized,
        "version": "optimized",
    }

    # --- 对比工具 ---
    _TOOL_REGISTRY["compare_stock_tools"] = {
        "skill": "__builtin__",
        "handler": _tool_compare_stock,
        "version": "compare",
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


def _normalize_date(date_str: str) -> str:
    """Normalize date formats like 20260804 -> 2026-08-04."""
    date_str = date_str.strip()
    # Already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    # Compact format YYYYMMDD
    m = re.match(r'^(\d{4})(\d{2})(\d{2})$', date_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Other common formats
    m = re.match(r'^(\d{4})/(\d{1,2})/(\d{1,2})$', date_str)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return date_str


def _tool_fetch_stock(arguments: dict) -> dict:
    company = arguments.get("company", "")
    date = _normalize_date(arguments.get("date", ""))
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
            cwd=script_dir, timeout=120, encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = stdout + stderr
        status = "completed" if result.returncode == 0 else "failed"

        generated_urls = _extract_urls_from_output("stock-dashboard", output)

        # Parse key metrics from output
        kline_count = 0
        verdict = "-"
        m = re.search(r'30分钟K线:\s*(\d+)', output)
        if m:
            kline_count = int(m.group(1))
        m = re.search(r'判定结果:\s*(\S+)', output)
        if m:
            verdict = m.group(1)

        return {
            "tool": "fetch_stock",
            "command": " ".join(cmd_parts),
            "status": status,
            "output": output[-3000:],
            "returncode": result.returncode,
            "generated_urls": generated_urls,
            "kline_count": kline_count,
            "verdict": verdict,
        }
    except subprocess.TimeoutExpired:
        return {"tool": "fetch_stock", "status": "timeout", "error": "执行超时(120s)"}
    except Exception as e:
        return {"tool": "fetch_stock", "status": "error", "error": str(e)}


def _tool_fetch_stock_optimized(arguments: dict) -> dict:
    """优化版：直接import模块调用，避免subprocess开销，返回结构化摘要。"""
    company = arguments.get("company", "")
    date = _normalize_date(arguments.get("date", ""))
    if not company or not date:
        return {"error": "缺少必要参数: company 和 date"}

    skill_path = skill_loader.get_skill_path("stock-dashboard-optimized")
    script_dir = os.path.join(skill_path, "scripts")

    # 直接import优化模块
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import fetch_stock_opt
        result_data = fetch_stock_opt.execute(
            company=company,
            date_str=date,
            from_json=arguments.get("from_json", False),
            skip_html=arguments.get("skip_html", False),
        )

        if result_data.get("status") == "error":
            return {
                "tool": "fetch_stock_optimized",
                "status": "failed",
                "error": result_data["message"],
                "generated_urls": [],
            }

        generated_urls = []
        if result_data.get("html_path"):
            url = _path_to_url("stock-dashboard-optimized", result_data["html_path"])
            generated_urls.append(url)
        if result_data.get("json_path"):
            url = _path_to_url("stock-dashboard-optimized", result_data["json_path"])
            generated_urls.append(url)

        return {
            "tool": "fetch_stock_optimized",
            "status": "completed",
            "company": result_data["company"],
            "stock_code": result_data["stock_code"],
            "date": result_data["date"],
            "verdict": result_data["verdict"],
            "kline_count": result_data["kline_count"],
            "buy_ratio": result_data["buy_ratio"],
            "sell_ratio": result_data["sell_ratio"],
            "buy_sell_ratio": result_data["buy_sell_ratio"],
            "pct_change": result_data.get("pct_change"),
            "elapsed_seconds": result_data["elapsed_seconds"],
            "generated_urls": generated_urls,
        }
    except ImportError as e:
        return {"tool": "fetch_stock_optimized", "status": "error", "error": f"导入失败: {e}"}
    except Exception as e:
        return {"tool": "fetch_stock_optimized", "status": "error", "error": str(e)}


def _tool_compare_stock(arguments: dict) -> dict:
    """对比工具：同时执行原版和优化版股票查询，返回对比结果。"""
    company = arguments.get("company", "")
    date = _normalize_date(arguments.get("date", ""))
    if not company or not date:
        return {"error": "缺少必要参数: company 和 date"}

    # --- 执行原版 ---
    orig_start = time.time()
    orig_result = _tool_fetch_stock({"company": company, "date": date})
    orig_elapsed = time.time() - orig_start

    # --- 执行优化版 ---
    opt_start = time.time()
    opt_result = _tool_fetch_stock_optimized({"company": company, "date": date})
    opt_elapsed = time.time() - opt_start

    # --- 构建对比结果 ---
    orig_ok = orig_result.get("status") == "completed"
    opt_ok = opt_result.get("status") == "completed"

    generated_urls = []
    for r in [orig_result, opt_result]:
        for url in r.get("generated_urls", []):
            generated_urls.append(url)

    speedup = f"{(orig_elapsed / opt_elapsed):.1f}x" if opt_elapsed > 0 else "N/A"

    # Estimate output token efficiency (based on output data size)
    orig_output_size = len(json.dumps(orig_result, ensure_ascii=False))
    opt_output_size = len(json.dumps(opt_result, ensure_ascii=False))
    output_reduction = f"{((1 - opt_output_size / orig_output_size) * 100):.0f}%" if orig_output_size > 0 else "N/A"

    return {
        "tool": "compare_stock_tools",
        "status": "completed",
        "company": company,
        "date": date,
        "comparison": {
            "original": {
                "tool_name": "fetch_stock (原版)",
                "status": orig_result.get("status", "error"),
                "elapsed_seconds": round(orig_elapsed, 2),
                "kline_count": orig_result.get("kline_count", 0),
                "verdict": orig_result.get("verdict", "-"),
                "error": orig_result.get("error"),
                "output_size": orig_output_size,
            },
            "optimized": {
                "tool_name": "fetch_stock_optimized (优化版)",
                "status": opt_result.get("status", "error"),
                "elapsed_seconds": round(opt_elapsed, 2),
                "kline_count": opt_result.get("kline_count", 0),
                "verdict": opt_result.get("verdict", "-"),
                "error": opt_result.get("error"),
                "output_size": opt_output_size,
            },
        },
        "speedup": speedup,
        "output_reduction": output_reduction,
        "generated_urls": generated_urls,
        "summary": (
            f"原版耗时 {orig_elapsed:.2f}s {'成功' if orig_ok else '失败'} | "
            f"优化版耗时 {opt_elapsed:.2f}s {'成功' if opt_ok else '失败'} | "
            f"加速比 {speedup} | 输出减少 {output_reduction}"
        ),
    }


def _tool_save_flashcard_data(arguments: dict) -> dict:
    """保存单词闪卡 JSON 数据到 flash-card/data/ 目录。"""
    word = arguments.get("word", "")
    if not word:
        return {"error": "缺少 word 参数"}

    skill_path = skill_loader.get_skill_path("flash-card")
    data_dir = os.path.join(skill_path, "data")
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
    with open(file_path, "w", encoding="locale") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    file_url = _path_to_url("flash-card", file_path)

    return {
        "tool": "save_flashcard_data",
        "status": "completed",
        "file": file_path,
        "file_url": file_url,
        "word": word,
    }


def _tool_generate_flashcard(arguments: dict) -> dict:
    """运行 make_flashcard.py 从 JSON 生成 HTML 闪卡。"""
    word = arguments.get("word", "")
    if not word:
        return {"error": "缺少 word 参数"}

    skill_path = skill_loader.get_skill_path("flash-card")
    data_file = os.path.join(skill_path, "data", f"{word.lower()}.json")

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
    """构建所有 Skills 的工具定义（精简版，降低token消耗）。"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "compare_stock_tools",
                "description": "【推荐】同时调用原版和优化版股票查询工具，返回执行时间、状态、结果的对比数据。用户查询股票时应优先使用此工具。日期格式支持 YYYY-MM-DD 或 YYYYMMDD。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string", "description": "A股公司名称，如 比亚迪、宁德时代"},
                        "date": {"type": "string", "description": "交易日期，支持 YYYY-MM-DD 或 YYYYMMDD 格式"},
                    },
                    "required": ["company", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_stock",
                "description": "【原版】获取股票30分钟K线并生成HTML看板。通过subprocess执行，输出详细日志。日期支持 YYYY-MM-DD 或 YYYYMMDD。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string", "description": "A股公司名称"},
                        "date": {"type": "string", "description": "交易日期，支持 YYYY-MM-DD 或 YYYYMMDD 格式"},
                        "from_json": {"type": "boolean", "description": "从已有JSON生成HTML"},
                        "skip_html": {"type": "boolean", "description": "跳过HTML生成"},
                    },
                    "required": ["company", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_stock_optimized",
                "description": "【优化版】获取股票30分钟K线并生成HTML看板。内置代码缓存、快速重试，直接模块调用，返回精简摘要。日期支持 YYYY-MM-DD 或 YYYYMMDD。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string", "description": "A股公司名称"},
                        "date": {"type": "string", "description": "交易日期，支持 YYYY-MM-DD 或 YYYYMMDD 格式"},
                        "from_json": {"type": "boolean", "description": "从已有JSON生成HTML"},
                        "skip_html": {"type": "boolean", "description": "跳过HTML生成"},
                    },
                    "required": ["company", "date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_all_skills",
                "description": "列出系统中所有可用的 Skill",
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
                        "skill_name": {"type": "string", "description": "Skill 名称"},
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

        chat_start = time.time()
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            final_reply, tool_records, skill_used, token_usage = self._run_tool_loop(
                messages, tools
            )
            total_tokens = token_usage
        except ValueError as e:
            return models.ChatResponse(
                reply=str(e),
                total_time_ms=int((time.time() - chat_start) * 1000),
                token_usage=total_tokens,
            )
        except RuntimeError as e:
            return models.ChatResponse(
                reply=f"LLM 调用失败: {e}",
                total_time_ms=int((time.time() - chat_start) * 1000),
                token_usage=total_tokens,
            )

        total_time_ms = int((time.time() - chat_start) * 1000)
        skill_info = skill_loader.get_skill(skill_used) if skill_used else None

        return models.ChatResponse(
            reply=final_reply,
            next_step=None,
            action_suggestion=None,
            skill_info=skill_info,
            skill_used=skill_used,
            tool_calls=tool_records,
            total_time_ms=total_time_ms,
            token_usage=total_tokens,
        )

    def _run_tool_loop(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[str, list[models.ToolCallRecord], Optional[str], dict]:
        tool_records: list[models.ToolCallRecord] = []
        skill_used: Optional[str] = None
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for round_idx in range(MAX_TOOL_ROUNDS):
            tool_choice = "required" if round_idx == 0 else None
            result = llm_client.chat(messages, tools=tools, tool_choice=tool_choice)

            # 累计token使用量
            usage = result.get("usage", {})
            total_tokens["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_tokens["completion_tokens"] += usage.get("completion_tokens", 0)
            total_tokens["total_tokens"] += usage.get("total_tokens", 0)

            assistant_msg = {"role": "assistant", "content": result["content"]}
            if result["tool_calls"]:
                assistant_msg["tool_calls"] = result["tool_calls"]
            messages.append(assistant_msg)

            if not result["tool_calls"]:
                return result["content"] or "(模型未生成回复)", tool_records, skill_used, total_tokens

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

                comparison_data = None
                if tool_name == "compare_stock_tools" and tool_result.get("comparison"):
                    comparison_data = tool_result["comparison"]
                    comparison_data["speedup"] = tool_result.get("speedup", "N/A")
                    comparison_data["token_usage"] = dict(total_tokens)

                tool_records.append(models.ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    status=tool_result.get("status", "completed"),
                    duration_ms=tool_duration,
                    summary=summary,
                    comparison=comparison_data,
                ))

                tool_reply = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                messages.append(tool_reply)

        last_msg = messages[-1]
        return last_msg.get("content", "") or "(达到最大工具调用轮次)", tool_records, skill_used, total_tokens

    def _summarize_tool_result(self, tool_name: str, result: dict, duration_ms: int) -> str:
        status = result.get("status", "completed")
        urls = result.get("generated_urls", [])

        # 对比工具特殊展示
        if tool_name == "compare_stock_tools" and result.get("comparison"):
            comp = result["comparison"]
            orig = comp.get("original", {})
            opt = comp.get("optimized", {})
            speedup = result.get("speedup", "N/A")
            output_reduction = result.get("output_reduction", "N/A")
            parts = [f"📊 对比完成（总耗时 {duration_ms}ms）"]
            parts.append(f"  原版: {orig.get('elapsed_seconds', '-')}s | 状态: {orig.get('status', '-')} | K线: {orig.get('kline_count', 0)}根 | 输出: {orig.get('output_size', 0)}字节")
            parts.append(f"  优化版: {opt.get('elapsed_seconds', '-')}s | 状态: {opt.get('status', '-')} | K线: {opt.get('kline_count', 0)}根 | 输出: {opt.get('output_size', 0)}字节")
            parts.append(f"  加速比: {speedup} | 输出减少: {output_reduction}")
            if urls:
                parts.append(f"  输出文件: {len(urls)} 个")
            return "\n".join(parts)

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
