"""
Skill 声明式加载器

设计要点：
  1. 每个 skill 独立目录：skills/{skill_name}/skill.md (+ 可选 .py 脚本)
  2. skill.md 使用 YAML frontmatter 声明元信息，正文为使用说明
  3. 按需加载：get_skill_catalog() 只返回 name+description，load_skill_detail() 返回完整定义
  4. 两种 executor：
     - script: subprocess 执行 .py，参数以 JSON 通过 stdin 传入，stdout 作为返回
     - llm: 纯描述型，返回 skill 描述供 LLM 自行处理

使用方式：
  from skill import get_skill_catalog, load_skill_detail, execute_skill

  # 获取 skill 目录（注入 system prompt）
  catalog = get_skill_catalog()

  # 按需加载完整定义
  detail = load_skill_detail("calculator")

  # 执行 skill
  result = execute_skill("calculator", {"expr": "123*456"})
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import yaml

logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SKILLS_DIR = Path(__file__).parent.parent / "skills"
SCRIPT_TIMEOUT = int(os.getenv("SKILL_SCRIPT_TIMEOUT", "30"))


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class SkillInfo:
    """单个 skill 的解析结果"""
    name: str
    description: str
    executor: str                       # "script" | "llm"
    script_path: Optional[Path] = None  # executor=script 时有效
    parameters: List[Dict] = field(default_factory=list)
    usage_doc: str = ""                 # MD 正文（使用说明）
    skill_dir: Path = None


# ── 全局缓存 ──────────────────────────────────────────────────────────────────
_skills_cache: Dict[str, SkillInfo] = {}


# ── MD 解析 ───────────────────────────────────────────────────────────────────

def _parse_skill_md(path: Path) -> Optional[SkillInfo]:
    """解析单个 skill.md 文件，返回 SkillInfo"""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"读取 skill.md 失败 {path}: {e}")
        return None

    # 分离 YAML frontmatter 和正文
    if not content.startswith("---"):
        logger.warning(f"skill.md 缺少 frontmatter: {path}")
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        logger.warning(f"skill.md frontmatter 格式错误: {path}")
        return None

    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        logger.warning(f"skill.md YAML 解析失败 {path}: {e}")
        return None

    usage_doc = parts[2].strip()

    name = meta.get("name") or path.parent.name
    description = meta.get("description", "")
    executor = meta.get("executor", "llm")
    script_name = meta.get("script")
    parameters = meta.get("parameters", [])

    script_path = None
    if executor == "script":
        if script_name:
            script_path = path.parent / script_name
        else:
            # 默认查找 {skill_name}.py
            script_path = path.parent / f"{name}.py"
        if not script_path.exists():
            logger.warning(f"skill '{name}' 的脚本不存在: {script_path}")
            script_path = None

    return SkillInfo(
        name=name,
        description=description,
        executor=executor,
        script_path=script_path,
        parameters=parameters or [],
        usage_doc=usage_doc,
        skill_dir=path.parent,
    )


# ── 目录扫描 ──────────────────────────────────────────────────────────────────

def scan_skills(skills_dir: Path = None) -> Dict[str, SkillInfo]:
    """扫描 skills/ 下的所有子目录，解析 skill.md"""
    skills_dir = skills_dir or SKILLS_DIR
    result = {}

    if not skills_dir.exists():
        logger.warning(f"skills 目录不存在: {skills_dir}")
        return result

    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "skill.md"
        if not skill_md.exists():
            continue

        info = _parse_skill_md(skill_md)
        if info:
            result[info.name] = info
            logger.debug(f"加载 skill: {info.name} (executor={info.executor})")

    return result


def reload() -> Dict[str, Any]:
    """热重载，重新扫描 skills 目录"""
    global _skills_cache
    _skills_cache = scan_skills()
    return {
        "count": len(_skills_cache),
        "names": list(_skills_cache.keys()),
    }


def _ensure_loaded():
    """确保 skills 已加载（懒初始化）"""
    global _skills_cache
    if not _skills_cache:
        _skills_cache = scan_skills()


# ── Catalog 与按需加载 ────────────────────────────────────────────────────────

def get_skill_catalog() -> List[Dict[str, str]]:
    """返回轻量目录 [{name, description}]，用于注入 system prompt"""
    _ensure_loaded()
    return [
        {"name": info.name, "description": info.description}
        for info in _skills_cache.values()
    ]


def load_skill_detail(skill_name: str) -> str:
    """
    加载单个 skill 的完整定义（参数 schema + 使用说明）

    返回文本格式，供 LLM 了解如何调用 execute_skill
    """
    _ensure_loaded()
    info = _skills_cache.get(skill_name)
    if not info:
        return f"错误：技能 '{skill_name}' 不存在。可用技能：{list(_skills_cache.keys())}"

    lines = [f"技能: {info.name}"]
    lines.append(f"描述: {info.description}")
    lines.append(f"执行器: {info.executor}")

    if info.parameters:
        lines.append("\n参数:")
        for param in info.parameters:
            req = "必填" if param.get("required") else "可选"
            default = f"，默认: {param.get('default')}" if "default" in param else ""
            lines.append(
                f"  - {param['name']} ({param.get('type', 'string')}, {req}): "
                f"{param.get('description', '')}{default}"
            )
    else:
        lines.append("\n参数: 无")

    if info.usage_doc:
        lines.append(f"\n使用说明:\n{info.usage_doc}")

    lines.append(
        f"\n调用方式: execute_skill(skill_name=\"{info.name}\", parameters={{...}})"
    )

    return "\n".join(lines)


def get_skill_info(skill_name: str) -> Optional[SkillInfo]:
    """获取 skill 的原始信息（内部使用）"""
    _ensure_loaded()
    return _skills_cache.get(skill_name)


# ── 执行 ──────────────────────────────────────────────────────────────────────

def execute_skill(skill_name: str, parameters: Dict[str, Any]) -> str:
    """
    执行指定 skill

    Args:
        skill_name: 技能名称
        parameters: 参数字典

    Returns:
        执行结果字符串
    """
    _ensure_loaded()
    info = _skills_cache.get(skill_name)

    if not info:
        return f"错误：技能 '{skill_name}' 不存在。可用技能：{list(_skills_cache.keys())}"

    if info.executor == "llm":
        # 纯描述型，返回使用说明供 LLM 自行处理
        return f"[LLM 型技能] {info.name}: {info.description}\n\n{info.usage_doc}"

    if info.executor == "script":
        if not info.script_path or not info.script_path.exists():
            return f"错误：技能 '{skill_name}' 的脚本不存在"

        return _run_script(info.script_path, parameters)

    return f"错误：未知 executor 类型 '{info.executor}'"


def _run_script(script_path: Path, parameters: Dict[str, Any]) -> str:
    """通过 subprocess 执行 Python 脚本，参数以 JSON 通过 stdin 传入"""
    try:
        stdin_data = json.dumps(parameters, ensure_ascii=False)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT,
            cwd=str(script_path.parent),
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()[:500] if result.stderr else "未知错误"
            return f"脚本执行失败 (exit={result.returncode}): {error_msg}"

        return result.stdout.strip() or "(脚本无输出)"

    except subprocess.TimeoutExpired:
        return f"脚本执行超时（{SCRIPT_TIMEOUT}秒）"
    except Exception as e:
        return f"脚本执行异常: {e}"


# ── Function Calling Schema ───────────────────────────────────────────────────

def get_load_skill_schema() -> Dict:
    """load_skill 工具的 JSON Schema"""
    return {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "加载指定技能的详细参数定义和使用说明。在调用 execute_skill 之前必须先调用此工具了解参数格式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要加载的技能名称",
                    },
                },
                "required": ["skill_name"],
            },
        },
    }


def get_execute_skill_schema() -> Dict:
    """execute_skill 工具的 JSON Schema"""
    return {
        "type": "function",
        "function": {
            "name": "execute_skill",
            "description": "执行指定技能。必须先调用 load_skill 了解参数格式后再调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要执行的技能名称",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "技能参数，键值对形式。具体参数请参考 load_skill 返回的定义。",
                    },
                },
                "required": ["skill_name", "parameters"],
            },
        },
    }


def list_skills_detail() -> List[Dict[str, Any]]:
    """列出所有 skill 的详细信息（用于 /skills API）"""
    _ensure_loaded()
    return [
        {
            "name": info.name,
            "description": info.description,
            "executor": info.executor,
            "has_script": info.script_path is not None and info.script_path.exists(),
            "parameters": info.parameters,
        }
        for info in _skills_cache.values()
    ]


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Skill 加载器")
    parser.add_argument("--detail", type=str, help="查看指定 skill 的完整定义")
    parser.add_argument("--exec", type=str, help="执行指定 skill")
    parser.add_argument("--params", type=str, default="{}", help="执行参数 (JSON)")
    args = parser.parse_args()

    if args.detail:
        print(load_skill_detail(args.detail))
    elif args.exec:
        params = json.loads(args.params)
        result = execute_skill(args.exec, params)
        print(f"结果: {result}")
    else:
        catalog = get_skill_catalog()
        print(f"\n已加载 {len(catalog)} 个 skill:")
        for s in catalog:
            print(f"  - {s['name']}: {s['description']}")
