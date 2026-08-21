"""
Harness Engineer
================
CLI 式问答 harness：
- 调用 deepseek-chat 模型进行对话
- 自动加载 skills/ 下的 skill 元信息
- 当用户请求匹配 skill 时，渐进式加载 SKILL.md 并通过工具调用执行 skill
- 会话记忆：自动保存/恢复对话历史，支持多会话管理

运行:
    python harness.py
    python harness.py -k <your_api_key>
    python harness.py --session my-session
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from openai import OpenAI, APIError
except ImportError:
    print("[harness] 缺少依赖 openai，请先执行: pip install openai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"

MAX_HISTORY_TURNS = 20
TRIM_KEEP_SYSTEM = True

EXIT_WORDS = {"exit", "quit", "q", "退出"}

CMD_PREFIXES = {"/save", "/load", "/sessions", "/new", "/forget", "/help", "/status"}

TOOL_LOAD_SKILL = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "当判断用户请求与某个 skill 相关时调用此函数以获取完整的 SKILL.md 内容 "
            "（包括触发场景、执行流程、数据格式等详细说明）。"
            "必须先调用 load_skill 了解 skill 的完整流程，再根据说明执行后续操作。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "要加载的 skill 名称，例如 'flash-card'。",
                }
            },
            "required": ["skill_name"],
        },
    },
}

TOOL_SAVE_JSON = {
    "type": "function",
    "function": {
        "name": "save_json_file",
        "description": (
            "把一段 JSON 文本写入指定路径。用于保存 skill 所需的数据文件 "
            "（例如 flash-card 所需的 data/<word>.json）。"
            "Skill 产物请保存到 output 目录下。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件的绝对路径或相对于当前工作目录的路径。",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的 JSON 字符串内容。",
                },
            },
            "required": ["path", "content"],
        },
    },
}

TOOL_RUN_SCRIPT = {
    "type": "function",
    "function": {
        "name": "run_script",
        "description": (
            "在本地执行一段 Python 脚本或命令，用于运行 skill 附带的脚本 "
            "（例如 make_flashcard.py）。执行结束后会返回 stdout 与 stderr。"
            "建议将输出产物重定向到 output 目录。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的完整命令，例如 'python path/to/script.py arg1 -o out.html'。",
                }
            },
            "required": ["command"],
        },
    },
}

TOOL_OPEN_FILE = {
    "type": "function",
    "function": {
        "name": "open_file",
        "description": (
            "使用系统默认程序打开一个文件（HTML 会用浏览器打开），用于让用户预览 skill 产物。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要打开的文件路径。",
                }
            },
            "required": ["path"],
        },
    },
}

TOOLS = [TOOL_LOAD_SKILL, TOOL_SAVE_JSON, TOOL_RUN_SCRIPT, TOOL_OPEN_FILE]


# ---------------------------------------------------------------------------
# Skill 解析
# ---------------------------------------------------------------------------
class SkillMeta:
    def __init__(self, name: str, path: Path, description: str, full_text: str):
        self.name = name
        self.path = path
        self.description = description
        self.full_text = full_text


_YAML_FENCE_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_skill_md(skill_md_path: Path) -> Optional[tuple[str, str]]:
    raw = skill_md_path.read_text(encoding="utf-8")
    m = _YAML_FENCE_RE.match(raw)
    if not m:
        return None
    frontmatter = m.group(1)
    name_match = re.search(r"^\s*name:\s*(.+?)\s*$", frontmatter, re.MULTILINE)
    desc_match = re.search(r"^\s*description:\s*(.+?)(?=\n\S|\Z)", frontmatter, re.DOTALL | re.MULTILINE)
    if not name_match:
        return None
    name = name_match.group(1).strip().strip('"').strip("'")
    description = ""
    if desc_match:
        description = desc_match.group(1).strip().strip('"').strip("'")
    return name, description


def discover_skills(skills_dir: Path) -> list[SkillMeta]:
    skills: list[SkillMeta] = []
    if not skills_dir.exists():
        return skills
    for sub in sorted(skills_dir.iterdir()):
        if not sub.is_dir():
            continue
        skill_md = sub / "SKILL.md"
        if not skill_md.exists():
            continue
        parsed = parse_skill_md(skill_md)
        name = parsed[0] if parsed else sub.name
        description = parsed[1] if parsed else ""
        full_text = skill_md.read_text(encoding="utf-8")
        skills.append(SkillMeta(name=name, path=sub, description=description, full_text=full_text))
    return skills


def build_skill_catalog(skills: list[SkillMeta]) -> str:
    if not skills:
        return "当前未发现任何 skill。"
    lines = ["可用 skills 列表："]
    for i, s in enumerate(skills, 1):
        lines.append(f"  {i}. {s.name} —— {s.description}")
    lines.append("当用户的请求与某个 skill 相关时，请先调用 load_skill 获取完整说明。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 会话存储
# ---------------------------------------------------------------------------
def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_") or "session"


def _session_file(sessions_dir: Path, session_id: str) -> Path:
    return sessions_dir / f"{_safe_filename(session_id)}.json"


def list_sessions(sessions_dir: Path) -> list[dict[str, Any]]:
    if not sessions_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for f in sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "id": data.get("id", f.stem),
                "name": data.get("name", f.stem),
                "updated_at": data.get("updated_at", ""),
                "turns": _count_turns(data.get("messages", [])),
                "file": str(f),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _count_turns(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def load_session(sessions_dir: Path, session_id: str) -> Optional[dict[str, Any]]:
    path = _session_file(sessions_dir, session_id)
    if not path.exists():
        for f in sessions_dir.glob("*.json"):
            if f.stem == session_id:
                path = f
                break
        else:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_session(sessions_dir: Path, session_id: str, name: str, model: str, messages: list[dict[str, Any]]) -> Path:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    path = _session_file(sessions_dir, session_id)
    existing = None
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None
    data = {
        "id": session_id,
        "name": name,
        "model": model,
        "created_at": existing["created_at"] if existing else now,
        "updated_at": now,
        "messages": messages,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def delete_session(sessions_dir: Path, session_id: str) -> bool:
    path = _session_file(sessions_dir, session_id)
    if path.exists():
        path.unlink()
        return True
    for f in sessions_dir.glob("*.json"):
        if f.stem == session_id:
            f.unlink()
            return True
    return False


def find_latest_session(sessions_dir: Path) -> Optional[dict[str, Any]]:
    sessions = list_sessions(sessions_dir)
    if not sessions:
        return None
    latest = sessions[0]
    return load_session(sessions_dir, latest["id"])


# ---------------------------------------------------------------------------
# 核心 Harness
# ---------------------------------------------------------------------------
class Harness:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        skills_dir: Path = DEFAULT_SKILLS_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        sessions_dir: Path = DEFAULT_SESSIONS_DIR,
        session_id: Optional[str] = None,
        no_memory: bool = False,
    ):
        if not api_key:
            raise ValueError("未检测到 API Key，请设置环境变量 DEEPSEEK_API_KEY 或使用 -k 参数。")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.skills_dir = skills_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.sessions_dir = sessions_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.skills = discover_skills(skills_dir)
        self._skill_index = {s.name: s for s in self.skills}
        self.no_memory = no_memory

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.messages: list[dict[str, Any]] = []
        self._system_prompt = ""
        self.session_id: str = ""
        self.session_name: str = ""
        self._dirty = False

        self._build_system_prompt()
        if session_id:
            self._load_or_create(session_id)
        elif not no_memory:
            latest = find_latest_session(self.sessions_dir)
            if latest:
                self._restore_from(latest)
            else:
                self._create_new_session()
        else:
            self._create_new_session()

    # ---------------- 系统提示 ----------------
    def _build_system_prompt(self) -> None:
        self._system_prompt = (
            "你是一个 Harness Engineer，能够进行日常问答，并在用户请求匹配某个 skill 时，"
            "通过工具调用执行 skill 流程。\n\n"
            f"{build_skill_catalog(self.skills)}\n\n"
            f"【重要路径信息】\n"
            f"- skills 根目录: {self.skills_dir}\n"
            f"- output 根目录（所有产物输出到此）: {self.output_dir}\n\n"
            "工作规则：\n"
            "1. 日常问答直接用文字回答即可，不要滥用工具。\n"
            "2. 当用户请求某个 skill 时，先调用 load_skill 拿到完整 SKILL.md 说明。\n"
            "3. 仔细阅读 skill 的执行流程、数据格式、注意事项，然后按步骤执行。\n"
            "4. 所有 skill 产物（HTML、JSON、脚本输出等）统一保存到 output 目录，"
            "save_json_file 的 path 参数、run_script 的命令、open_file 的 path 参数都应指向 output 目录。\n"
            "5. 执行完成后用自然语言向用户汇报产物位置（相对于 output 目录）和结果。"
        )

    def _inject_system_prompt(self) -> None:
        self.messages.insert(0, {"role": "system", "content": self._system_prompt})

    # ---------------- 会话管理 ----------------
    def _create_new_session(self) -> None:
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.session_name = self.session_id
        self.messages = []
        self._inject_system_prompt()
        self._dirty = True

    def _restore_from(self, data: dict[str, Any]) -> None:
        self.session_id = data.get("id", "")
        self.session_name = data.get("name", self.session_id)
        raw_messages = data.get("messages", [])
        self.messages = []
        system_found = False
        for m in raw_messages:
            role = m.get("role", "")
            if role == "system":
                system_found = True
                continue
            self.messages.append(m)
        self._inject_system_prompt()
        if not system_found:
            pass
        self._dirty = False

    def _load_or_create(self, session_id: str) -> None:
        data = load_session(self.sessions_dir, session_id)
        if data:
            self._restore_from(data)
        else:
            print(f"[session] 未找到 '{session_id}'，创建新会话。")
            self._create_new_session()

    def _persist(self) -> None:
        if self.no_memory or not self._dirty or not self.session_id:
            return
        try:
            path = save_session(
                self.sessions_dir,
                self.session_id,
                self.session_name,
                self.model,
                self.messages,
            )
            self._dirty = False
        except OSError as e:
            print(f"[session] 保存失败: {e}", file=sys.stderr)

    def _trim_history(self) -> None:
        if self.no_memory:
            return
        turns = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if len(turns) <= MAX_HISTORY_TURNS:
            return
        cutoff_idx = turns[-MAX_HISTORY_TURNS]
        new_messages: list[dict[str, Any]] = []
        recent = self.messages[cutoff_idx:]
        keep_ids: set[str] = set()
        for m in recent:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls", []) or []:
                    keep_ids.add(tc["id"])
        for m in self.messages:
            role = m.get("role", "")
            if role == "system":
                new_messages.append(m)
                continue
            if m not in recent:
                continue
            if role == "tool":
                if m.get("tool_call_id", "") in keep_ids:
                    new_messages.append(m)
            elif role == "assistant":
                tcs = m.get("tool_calls")
                if tcs:
                    filtered = [tc for tc in tcs if tc["id"] in keep_ids]
                    if filtered:
                        new_messages.append({**m, "tool_calls": filtered})
                else:
                    new_messages.append(m)
            elif role == "user":
                new_messages.append(m)
        self.messages = new_messages

    # ---------------- 工具实现 ----------------
    def _resolve_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        cwd = Path.cwd()
        candidate = (cwd / p).resolve()
        if candidate.exists():
            return candidate
        out_candidate = (self.output_dir / p).resolve()
        if out_candidate.exists():
            return out_candidate
        return candidate

    def _tool_load_skill(self, skill_name: str) -> str:
        skill = self._skill_index.get(skill_name)
        if not skill:
            available = ", ".join(self._skill_index.keys()) or "(无)"
            return f"未找到 skill '{skill_name}'。可用: {available}"
        return (
            f"[已加载 skill: {skill_name}，根目录: {skill.path}]\n\n"
            f"--- SKILL.md 完整内容 ---\n{skill.full_text}\n--- END ---"
        )

    def _tool_save_json_file(self, path: str, content: str) -> str:
        try:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = json.loads(content)
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return f"已保存 JSON 到 {target}"
        except json.JSONDecodeError as e:
            return f"JSON 解析失败: {e}"
        except OSError as e:
            return f"写入失败: {e}"

    def _tool_run_script(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=Path.cwd(),
                timeout=60,
            )
            parts = []
            if result.stdout:
                parts.append(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                parts.append(f"STDERR:\n{result.stderr.strip()}")
            if result.returncode != 0:
                parts.append(f"[退出码: {result.returncode}]")
            return "\n".join(parts) or "(命令无输出)"
        except subprocess.TimeoutExpired:
            return "命令执行超时 (>60s)"
        except Exception as e:
            return f"执行失败: {e}"

    def _tool_open_file(self, path: str) -> str:
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return f"文件不存在: {target}"
            import webbrowser
            webbrowser.open(target.as_uri())
            return f"已用默认程序打开: {target}"
        except Exception as e:
            return f"打开失败: {e}"

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "load_skill":
            return self._tool_load_skill(args.get("skill_name", ""))
        if name == "save_json_file":
            return self._tool_save_json_file(args.get("path", ""), args.get("content", ""))
        if name == "run_script":
            return self._tool_run_script(args.get("command", ""))
        if name == "open_file":
            return self._tool_open_file(args.get("path", ""))
        return f"未知工具: {name}"

    # ---------------- LLM 对话循环 ----------------
    def _chat_completion(self) -> dict[str, Any]:
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

    def _process_response(self, response) -> None:
        message = response.choices[0].message
        assistant_content = message.content or ""

        tool_calls = getattr(message, "tool_calls", None) or []

        if assistant_content:
            self.messages.append({"role": "assistant", "content": assistant_content})
            print(f"\n{assistant_content}")
            self._dirty = True

        for tc in tool_calls:
            fn = tc.function
            args_raw = fn.arguments or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
            print(f"\n[tool call] {fn.name}({json.dumps(args, ensure_ascii=False)})")
            result = self._dispatch_tool(fn.name, args)
            print(f"[tool result]\n{result}")

            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": fn.name, "arguments": args_raw},
                        }
                    ],
                }
            )
            self.messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
            self._dirty = True

        if tool_calls:
            sub_resp = self._chat_completion()
            self._process_response(sub_resp)

    def chat(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        self._dirty = True
        try:
            self._trim_history()
            response = self._chat_completion()
            self._process_response(response)
            self._persist()
        except APIError as e:
            print(f"\n[API 错误] {e}", file=sys.stderr)
            self.messages.pop()
            self._dirty = False
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)
            self.messages.pop()
            self._dirty = False

    # ---------------- 会话命令 ----------------
    def _handle_command(self, user_input: str) -> bool:
        lower = user_input.lower().strip()
        if lower == "/help":
            self._print_help()
            return True
        if lower == "/status":
            self._print_status()
            return True
        if lower == "/new":
            self._persist()
            self._create_new_session()
            print(f"[session] 已创建新会话: {self.session_name}  (id={self.session_id})")
            return True
        if lower == "/forget":
            self.messages = []
            self._inject_system_prompt()
            self._dirty = True
            print("[session] 已清空当前会话上下文（历史文件保留）。")
            return True
        if lower == "/sessions":
            self._list_sessions_cmd()
            return True
        if lower.startswith("/save"):
            parts = user_input.strip().split(maxsplit=1)
            name = parts[1] if len(parts) > 1 else self.session_name
            self.session_name = name
            self._persist()
            print(f"[session] 已保存会话 '{self.session_name}' (id={self.session_id})")
            return True
        if lower.startswith("/load"):
            parts = user_input.strip().split(maxsplit=1)
            if len(parts) < 2:
                print("用法: /load <session_id>")
                return True
            sid = parts[1].strip()
            self._persist()
            data = load_session(self.sessions_dir, sid)
            if data:
                self._restore_from(data)
                print(f"[session] 已加载会话: {self.session_name} (id={self.session_id})")
            else:
                available = list_sessions(self.sessions_dir)
                print(f"[session] 未找到 '{sid}'。可用会话:")
                for s in available[:10]:
                    print(f"  {s['id']}  {s['name']}  ({s['turns']} turns)")
            return True
        if lower.startswith("/delete"):
            parts = user_input.strip().split(maxsplit=1)
            if len(parts) < 2:
                print("用法: /delete <session_id>")
                return True
            sid = parts[1].strip()
            ok = delete_session(self.sessions_dir, sid)
            if ok:
                if sid == self.session_id:
                    self._create_new_session()
                print(f"[session] 已删除会话: {sid}")
            else:
                print(f"[session] 未找到会话: {sid}")
            return True
        return False

    def _print_help(self) -> None:
        print(
            "\n可用命令：\n"
            "  /help           显示此帮助\n"
            "  /status         显示当前会话信息\n"
            "  /sessions       列出所有保存的会话\n"
            "  /load <id>      加载指定会话\n"
            "  /save [name]    保存当前会话（可指定名称）\n"
            "  /new            创建全新会话\n"
            "  /forget         清空当前上下文（保留历史文件）\n"
            "  /delete <id>    删除指定会话\n"
        )

    def _print_status(self) -> None:
        turns = _count_turns(self.messages)
        print(
            f"\n当前会话:\n"
            f"  ID:    {self.session_id}\n"
            f"  名称:  {self.session_name}\n"
            f"  模型:  {self.model}\n"
            f"  轮数:  {turns}\n"
            f"  内存:  {len(self.messages)} messages\n"
            f"  路径:  {self.sessions_dir}\n"
            f"  状态:  {'已修改，待保存' if self._dirty else '已同步'}"
        )

    def _list_sessions_cmd(self) -> None:
        sessions = list_sessions(self.sessions_dir)
        if not sessions:
            print("[session] 暂无保存的会话。")
            return
        print("\n已保存会话：")
        current = self.session_id
        for s in sessions:
            mark = "  <当前>" if s["id"] == current else ""
            print(f"  {s['id']}  {s['name']}  ({s['turns']} turns)  {s['updated_at']}{mark}")
        print("\n使用 /load <id> 加载会话。")

    # ---------------- 交互式 CLI ----------------
    def repl(self) -> None:
        turns = _count_turns(self.messages)
        print("=" * 60)
        print("  Harness Engineer  |  deepseek-chat CLI")
        print(f"  Skill: {len(self.skills)} 个  |  Output: {self.output_dir}")
        if self.no_memory:
            print(f"  会话记忆: 已禁用（--no-memory）")
        else:
            print(f"  会话: {self.session_name} (id={self.session_id}, {turns} turns)")
        print(f"  输入 /help 查看命令  |  quit 退出")
        print("=" * 60)
        while True:
            try:
                user_input = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._persist()
                print("\nbye.")
                break
            if not user_input:
                continue
            if user_input.lower() in EXIT_WORDS:
                self._persist()
                print("bye.")
                break
            if user_input.startswith("/"):
                handled = self._handle_command(user_input)
                if handled:
                    continue
            self.chat(user_input)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harness Engineer: deepseek-chat + skills CLI"
    )
    parser.add_argument(
        "-k", "--api-key",
        default=DEFAULT_API_KEY,
        help="deepseek API key，默认读取 DEEPSEEK_API_KEY 环境变量",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="API base URL"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名")
    parser.add_argument(
        "--skills-dir",
        default=str(DEFAULT_SKILLS_DIR),
        help="skills 目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="产物输出目录",
    )
    parser.add_argument(
        "--sessions-dir",
        default=str(DEFAULT_SESSIONS_DIR),
        help="会话存储目录",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="指定要加载的会话 ID",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="禁用会话记忆（不加载历史、不保存）",
    )
    args = parser.parse_args()

    skills_dir = Path(args.skills_dir)
    output_dir = Path(args.output_dir)
    sessions_dir = Path(args.sessions_dir)
    harness = Harness(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        skills_dir=skills_dir,
        output_dir=output_dir,
        sessions_dir=sessions_dir,
        session_id=args.session,
        no_memory=args.no_memory,
    )
    harness.repl()


if __name__ == "__main__":
    main()