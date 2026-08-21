import os
import re
from pathlib import Path
from typing import Optional

from models import SkillInfo, SkillStep

SKILLS_DIR = os.environ.get("SKILLS_DIR", os.path.join(os.path.dirname(__file__), "..", "skills"))


def _parse_skill_md(md_path: Path) -> Optional[SkillInfo]:
    text = md_path.read_text(encoding="utf-8")

    meta_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    name = ""
    description = ""
    if meta_match:
        meta = meta_match.group(1)
        m = re.search(r'name:\s*"(.+?)"', meta)
        if m:
            name = m.group(1)
        m = re.search(r'description:\s*"(.+?)"', meta)
        if m:
            description = m.group(1)

    if not name:
        folder_name = md_path.parent.name
        name = folder_name

    steps: list[SkillStep] = []
    section_text = text

    progressive_match = re.search(
        r"渐进式执行步骤[^（]*[（(]?\s*Agent Side\s*[)）]?\s*\n(.*?)(?=\n## |\Z)",
        section_text,
        re.DOTALL,
    )
    if progressive_match:
        steps_text = progressive_match.group(1)
        step_pattern = re.compile(
            r"(\d+)\.\s*\*\*(.+?)\*\*[：:]\s*(.*?)(?=\n\d+\.\s*\*\*|\Z)",
            re.DOTALL,
        )
        for m in step_pattern.finditer(steps_text):
            step_num = int(m.group(1))
            title = m.group(2).strip()
            desc = m.group(3).strip()
            cmd_match = re.search(r"运行\s*`([^`]+)`", desc)
            command = cmd_match.group(1) if cmd_match else None
            steps.append(SkillStep(
                step_num=step_num,
                title=title,
                description=desc,
                command=command,
            ))

    params: dict = {}
    params_section_match = re.search(
        r"## 参数\s*\n\|[^|]+\|.*?\n\|.*?\n(.*?)(?=\n## |\Z)",
        text,
        re.DOTALL,
    )
    if params_section_match:
        params_text = params_section_match.group(1)
        for line in params_text.strip().split("\n"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) >= 2 and cols[0] and cols[0] != "---":
                params[cols[0]] = {"description": cols[1] if len(cols) > 1 else "", "example": cols[2] if len(cols) > 2 else ""}

    return SkillInfo(
        name=name,
        description=description,
        path=str(md_path.parent),
        steps=steps,
        parameters=params,
    )


def list_skills() -> list[SkillInfo]:
    skills_dir = Path(SKILLS_DIR)
    skills = []
    if not skills_dir.exists():
        return skills
    for item in sorted(skills_dir.iterdir()):
        if item.is_dir():
            skill_md = item / "SKILL.md"
            if skill_md.exists():
                info = _parse_skill_md(skill_md)
                if info:
                    skills.append(info)
    return skills


def get_skill(skill_name: str) -> Optional[SkillInfo]:
    skills_dir = Path(SKILLS_DIR)
    skill_dir = skills_dir / skill_name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    return _parse_skill_md(skill_md)


def read_skill_md(skill_name: str) -> Optional[str]:
    skills_dir = Path(SKILLS_DIR)
    skill_dir = skills_dir / skill_name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None
    return skill_md.read_text(encoding="utf-8")


def get_skill_path(skill_name: str) -> Optional[str]:
    skills_dir = Path(SKILLS_DIR)
    skill_dir = skills_dir / skill_name
    if not skill_dir.exists():
        return None
    return str(skill_dir)
