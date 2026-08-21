import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

DANGEROUS_PATTERNS = [
    r'(^|\s)(rm|rmdir)\s+(-rf\s+)?/?(\s|$|\*)',
    r'(^|\s)dd\s+if=',
    r'(^|\s)mkfs\.',
    r'(^|\s)fdisk\s+',
    r'(^|\s)parted\s+',
    r'(^|\s)mount\s+',
    r'(^|\s)umount\s+',
    r'(^|\s)mkswap\s+',
    r'(^|\s)swapoff\s+',
    r'(^|\s)swapon\s+',
    r'(^|\s)shutdown(\s|$)',
    r'(^|\s)reboot(\s|$)',
    r'(^|\s)halt(\s|$)',
    r'(^|\s)poweroff(\s|$)',
    r'(^|\s)init(\s|$)',
    r'(^|\s)init\s+[0-9]',
    r'(^|\s)systemctl\s+(reboot|poweroff|halt|shutdown)',
    r'(^|\s)chmod\s+777\s+/',
    r'(^|\s)chown\s+.*\s+/$',
    r'(^|\s)mv\s+/\s+',
    r'>\s+/dev/',
    r'\|\s*bash\s*$',
    r'\|\s*sh\s*$',
    r'</dev/',
    r'(^|\s):\(\)\s*\{',
    r'(^|\s)wget\s+.*\|',
    r'(^|\s)curl\s+.*\|',
    r'(^|\s)sudo\s+',
    r'(^|\s)passwd\s+',
    r'(^|\s)kill\s+-9\s+',
    r'(^|\s)pkill\s+-9\s+',
    r'(^|\s)rmmod\s+',
    r'(^|\s)modprobe\s+',
    r'(^|\s)insmod\s+',
    r'(^|\s)ifconfig\s+\w+\s+(down|up)',
    r'(^|\s)ip\s+link\s+set\s+\w+\s+(down|up)',
    r'(^|\s)dd$',
]

SYSTEM_PROTECTED_DIRS = [
    '/etc',
    '/bin',
    '/sbin',
    '/usr',
    '/boot',
    '/dev',
    '/proc',
    '/sys',
    '/lib',
    '/lib64',
    '/var/log',
    '/var/lib',
    '/root',
]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory inside the project. Cannot create directories in system paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to create (relative to project root)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file inside the project. Cannot write to system paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root)"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file from the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to project root)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command and return its stdout and stderr. Dangerous commands (rm -rf /, dd, mkfs, sudo, shutdown, etc.) are automatically blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
]


def is_dangerous_command(command: str) -> Optional[str]:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command.strip()):
            return f"命令被安全策略拦截（匹配危险模式: {pattern}）"
    return None


def is_system_path(resolved: Path) -> bool:
    resolved_str = str(resolved.resolve())
    for sys_dir in SYSTEM_PROTECTED_DIRS:
        if resolved_str == sys_dir or resolved_str.startswith(sys_dir + "/"):
            return True
    return False


class ToolExecutor:
    def __init__(
        self,
        project_root: str,
        confirm_commands: bool = True,
        allow_tools: bool = True,
        confirmation_handler: Optional[Callable[[str], bool]] = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.confirm_commands = confirm_commands
        self.allow_tools = allow_tools
        self.confirmation_handler = confirmation_handler

    def create_directory(self, path: str) -> Dict[str, Any]:
        full_path = self._resolve(path)
        if is_system_path(full_path):
            return {"success": False, "error": f"拒绝操作：不允许在系统目录创建目录 {full_path}"}
        full_path.mkdir(parents=True, exist_ok=True)
        return {"success": True, "message": f"目录已创建: {full_path}"}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        full_path = self._resolve(path)
        if is_system_path(full_path):
            return {"success": False, "error": f"拒绝操作：不允许写入系统路径 {full_path}"}
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return {"success": True, "message": f"文件已写入: {full_path} ({len(content)} 字符)"}

    def read_file(self, path: str) -> Dict[str, Any]:
        full_path = self._resolve(path)
        if not full_path.exists():
            return {"success": False, "error": f"文件不存在: {full_path}"}
        content = full_path.read_text(encoding="utf-8")
        return {"success": True, "content": content, "path": str(full_path)}

    def execute_command(self, command: str, timeout: float = 30) -> Dict[str, Any]:
        danger = is_dangerous_command(command)
        if danger:
            return {"success": False, "error": danger, "stdout": "", "stderr": ""}

        if self.confirmation_handler:
            approved = self.confirmation_handler(command)
            if not approved:
                return {"success": False, "error": "用户拒绝执行该命令", "stdout": "", "stderr": ""}
        elif self.confirm_commands:
            sys.stderr.write(f"\n⚠️  即将执行命令: {command}\n")
            resp = input("确认执行? [Y/n] ").strip().lower()
            if resp not in ("", "y", "yes"):
                return {"success": False, "error": "用户取消执行", "stdout": "", "stderr": ""}

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root),
            )
            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout[:8000],
                "stderr": result.stderr[:2000],
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"命令执行超时 ({timeout}s)", "stdout": "", "stderr": ""}
        except Exception as e:
            return {"success": False, "error": str(e), "stdout": "", "stderr": ""}

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.allow_tools:
            return {"success": False, "error": "工具调用已禁用（需 --allow-tools）"}
        method = getattr(self, name, None)
        if not method:
            return {"success": False, "error": f"未知工具: {name}"}
        try:
            return method(**arguments)
        except Exception as e:
            return {"success": False, "error": str(e)}


_executor: Optional[ToolExecutor] = None
_skills: Optional[Dict[str, Any]] = None
_skill_prompt: str = ""


def _load_skills():
    global _skills, _skill_prompt
    if _skills is not None:
        return
    try:
        from skills import discover_skills, get_skill_prompts
        _skills = discover_skills()
        _skill_prompt = get_skill_prompts(_skills)
        if _skills:
            names = ", ".join(_skills.keys())
            logging.getLogger(__name__).info(f"已加载技能: {names}")
    except Exception as e:
        _skills = {}
        _skill_prompt = ""
        logging.getLogger(__name__).debug(f"技能加载跳过: {e}")


def get_all_tools() -> list:
    _load_skills()
    if not _skills:
        return TOOL_DEFINITIONS
    from skills import get_skill_tools
    return TOOL_DEFINITIONS + get_skill_tools(_skills)


def get_skill_descriptions() -> str:
    _load_skills()
    return _skill_prompt


def get_tool_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        _executor = ToolExecutor(project_root=str(Path(__file__).parent.parent), allow_tools=False)
    return _executor


def set_tool_executor(executor: ToolExecutor):
    global _executor
    _executor = executor


# Patch ToolExecutor.execute_tool to also check skills
_orig_execute_tool = ToolExecutor.execute_tool


def _patched_execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not self.allow_tools:
        return {"success": False, "error": "工具调用已禁用（需 --allow-tools）"}
    method = getattr(self, name, None)
    if method:
        try:
            return method(**arguments)
        except Exception as e:
            return {"success": False, "error": str(e)}
    _load_skills()
    if _skills and name in _skills:
        try:
            return _skills[name].execute(name, arguments)
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "error": f"未知工具: {name}"}


ToolExecutor.execute_tool = _patched_execute_tool
