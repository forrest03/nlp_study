"""技能注册中心：自动发现 skills/ 下的技能"""
import importlib
import re
from pathlib import Path
from typing import Dict, List, Optional

from skills.base import Skill

SKILLS_DIR = Path(__file__).parent


def _parse_frontmatter(file_path: Path) -> Optional[dict]:
    """解析 Markdown 文件开头的 YAML-like 前置元数据（--- 包围的块）。

    支持单行：key: value
    支持多行：key: |\n  value（缩进的行会被拼接）
    """
    if not file_path.exists():
        return None
    text = file_path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    meta = {}
    lines = m.group(1).splitlines()
    current_key = None
    ml_indent = None
    for line in lines:
        stripped = line.strip()
        if current_key is not None:
            if stripped == "":
                meta[current_key] += "\n"
                continue
            leading = line[:len(line) - len(line.lstrip())]
            if ml_indent is not None and leading.startswith(ml_indent):
                meta[current_key] += "\n" + stripped
                continue
            else:
                # 多行块结束，回退处理当前行
                current_key = None
                ml_indent = None
        if current_key is None and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "|":
                current_key = key
                ml_indent = " " * (len(line) - len(line.lstrip()))
                meta[key] = ""
            else:
                meta[key] = val
    return meta if meta else None


def discover_skills() -> Dict[str, Skill]:
    skills: Dict[str, Skill] = {}
    for entry in SKILLS_DIR.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        mod_path = entry / "__init__.py"
        if not mod_path.exists():
            continue
        try:
            mod = importlib.import_module(f"skills.{entry.name}")
            if not (hasattr(mod, "skill") and isinstance(mod.skill, Skill)):
                continue
            sk = mod.skill

            # 从 .md 文件前置元数据读取 name / description，覆盖代码中的值
            md_files = list(entry.glob("*.md"))
            for mf in md_files:
                meta = _parse_frontmatter(mf)
                if meta:
                    if "name" in meta:
                        sk.name = meta["name"]
                    if "description" in meta:
                        sk.description = meta["description"]

            skills[sk.name] = sk
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"加载技能 {entry.name} 失败: {e}")
    return skills


def get_skill_tools(skills: Dict[str, Skill]) -> List[dict]:
    tools = []
    for s in skills.values():
        tools.extend(s.tools)
    return tools


def get_skill_prompts(skills: Dict[str, Skill]) -> str:
    parts = []
    for s in skills.values():
        if s.description:
            parts.append(s.description)
    if parts:
        return "\n\n".join(parts)
    return ""
