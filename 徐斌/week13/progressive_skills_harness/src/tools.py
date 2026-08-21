"""
Harness 工具集：渐进加载 + 文件/脚本执行。

教学点：Skill 本身不是 Function Call schema；
Function Call 是 Harness 提供的「加载与执行原语」，Skill 按需指挥这些原语。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from progressive_loader import ProgressiveLoader
from skill_registry import SkillRegistry


def _safe_workspace_path(workspace: Path, relative: str) -> Path:
    rel = relative.lstrip("/").replace("\\", "/")
    if ".." in rel.split("/"):
        raise ValueError("path traversal denied")
    target = (workspace / rel).resolve()
    target.relative_to(workspace.resolve())
    return target


def build_tools(
    registry: SkillRegistry,
    loader: ProgressiveLoader,
    workspace: Path,
    project_root: Path,
) -> tuple[list[dict], dict[str, Callable[..., Any]]]:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    project_root = Path(project_root)

    def list_skills() -> dict:
        hints = []
        for m in registry.list_metas():
            hints.append(
                {
                    "name": m.name,
                    "description": m.description[:200],
                    "triggers": m.triggers,
                    "base_dir": str(m.path.resolve()),
                }
            )
        snap = loader.snapshot()
        return {"skills": hints, "token_snapshot": {
            "l0": snap["l0_tokens"],
            "full_load": snap["full_load_tokens"],
        }}

    def activate_skill(name: str) -> dict:
        return loader.activate(name)

    def read_skill_file(skill: str, path: str) -> dict:
        return loader.read_resource(skill, path)

    def release_skill(name: str | None = None) -> dict:
        return loader.release(name)

    def write_file(path: str, content: str) -> dict:
        """写入文件。相对路径默认落在 workspace/；也允许写入已激活 skill 目录。"""
        path = path.replace("\\", "/")
        # 允许写入 skill 内 data/
        if path.startswith("skills/"):
            target = (project_root / path).resolve()
            try:
                target.relative_to(project_root.resolve())
            except ValueError:
                return {"ok": False, "error": "outside project"}
        else:
            # workspace 相对路径；兼容传入 workspace/xxx
            rel = path[len("workspace/") :] if path.startswith("workspace/") else path
            try:
                target = _safe_workspace_path(workspace, rel)
            except ValueError as e:
                return {"ok": False, "error": str(e)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(target)}

    def read_file(path: str) -> dict:
        path = path.replace("\\", "/")
        candidates = []
        if path.startswith("skills/"):
            candidates.append(project_root / path)
        elif path.startswith("workspace/"):
            candidates.append(workspace / path[len("workspace/") :])
        else:
            candidates.append(workspace / path)
            candidates.append(project_root / path)
        for c in candidates:
            c = c.resolve()
            if c.exists() and c.is_file():
                # 限制只能读 project / workspace
                try:
                    c.relative_to(project_root.resolve())
                except ValueError:
                    try:
                        c.relative_to(workspace.resolve())
                    except ValueError:
                        return {"ok": False, "error": "path not allowed"}
                text = c.read_text(encoding="utf-8")
                if len(text) > 80_000:
                    text = text[:80_000] + "\n...[truncated]..."
                return {"ok": True, "path": str(c), "content": text}
        return {"ok": False, "error": f"file not found: {path}"}

    def run_skill_script(command: str, timeout_sec: int = 60) -> dict:
        """
        在 project_root 下执行 shell 命令（教学场景）。
        仅允许 python / 相对脚本路径；禁止危险模式。
        """
        banned = ["rm -rf", "sudo ", "$(", "`", "curl ", "wget "]
        lowered = command.lower()
        for b in banned:
            if b in lowered:
                return {"ok": False, "error": f"command contains banned pattern: {b.strip()}"}

        cwd = str(project_root.resolve())
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout_sec}s"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

        stdout = (proc.stdout or "")[-8000:]
        stderr = (proc.stderr or "")[-4000:]
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "python": sys.executable,
        }

    def get_load_stats() -> dict:
        return loader.snapshot()

    tools_map: dict[str, Callable[..., Any]] = {
        "list_skills": lambda **_: list_skills(),
        "activate_skill": lambda name, **_: activate_skill(name),
        "read_skill_file": lambda skill, path, **_: read_skill_file(skill, path),
        "release_skill": lambda name=None, **_: release_skill(name),
        "write_file": lambda path, content, **_: write_file(path, content),
        "read_file": lambda path, **_: read_file(path),
        "run_skill_script": lambda command, timeout_sec=60, **_: run_skill_script(
            command, timeout_sec=int(timeout_sec)
        ),
        "get_load_stats": lambda **_: get_load_stats(),
    }

    schema = [
        {
            "type": "function",
            "function": {
                "name": "list_skills",
                "description": "列出所有 Skill 的摘要（L0）。通常 system 已含索引，仅在需要刷新时调用。",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "activate_skill",
                "description": "渐进式加载：将完整 SKILL.md 注入上下文（L1）。执行某 skill 前必须先激活。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill 名称，如 flash-card"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_skill_file",
                "description": "按需加载 Skill 内部资源（L2），如 references/architecture.md。须先 activate。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "path": {"type": "string", "description": "相对 skill 根目录的路径"},
                    },
                    "required": ["skill", "path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "release_skill",
                "description": "任务完成后释放已激活 Skill，节省后续轮次 context。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "省略则释放全部"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "写入文本文件。相对路径默认写入 workspace/；也可写 skills/<name>/data/...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取 workspace 或项目内文本文件。",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_skill_script",
                "description": "在项目根目录执行 shell 命令（如 python skills/flash-card/scripts/make_flashcard.py ...）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_sec": {"type": "integer", "default": 60},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_load_stats",
                "description": "查看当前渐进式加载的 token 占用与事件（教学用）。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    return schema, tools_map


def dispatch(tools_map: dict[str, Callable[..., Any]], name: str, arguments: str | dict) -> str:
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}
    else:
        args = arguments or {}
    fn = tools_map.get(name)
    if fn is None:
        return json.dumps({"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False)
    try:
        result = fn(**args)
    except TypeError as e:
        result = {"ok": False, "error": f"bad arguments: {e}"}
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    return json.dumps(result, ensure_ascii=False, default=str)
