from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path

    @property
    def directory(self) -> Path:
        return self.path.parent


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_skill_file(path: Path) -> Skill:
    frontmatter: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "---":
            raise ValueError(f"{path} 缺少 YAML frontmatter")
        for line in handle:
            if line.strip() == "---":
                break
            frontmatter.append(line.rstrip("\r\n"))
        else:
            raise ValueError(f"{path} 的 YAML frontmatter 未闭合")

    metadata: dict[str, str] = {}
    for line in frontmatter:
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _unquote(value)

    name = metadata.get("name", "").strip()
    description = metadata.get("description", "").strip()
    if not name or not description:
        raise ValueError(f"{path} 必须包含 name 和 description")

    return Skill(
        name=name,
        description=description,
        path=path.resolve(),
    )


class SkillRegistry:
    def __init__(self, skills: Iterable[Skill]):
        self._skills = {skill.name: skill for skill in skills}

    @classmethod
    def discover(cls, root: Path) -> "SkillRegistry":
        if not root.exists():
            raise FileNotFoundError(f"Skills 目录不存在：{root}")
        skills = [parse_skill_file(path) for path in sorted(root.glob("*/SKILL.md"))]
        if not skills:
            raise ValueError(f"没有在 {root} 中发现 Skill")
        return cls(skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def require(self, name: str) -> Skill:
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"未知 Skill：{name}")
        return skill

    def catalog_text(self) -> str:
        lines = ["可用 Skills（这里只是元数据，尚未加载完整指令）："]
        for name in self.names():
            skill = self._skills[name]
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def load_body(self, skill: Skill) -> str:
        with skill.path.open("r", encoding="utf-8") as handle:
            if handle.readline().strip() != "---":
                raise ValueError(f"{skill.path} 缺少 YAML frontmatter")
            for line in handle:
                if line.strip() == "---":
                    return handle.read().strip()
        raise ValueError(f"{skill.path} 的 YAML frontmatter 未闭合")

    def read_reference(self, skill: Skill, relative_path: str) -> str:
        requested = (skill.directory / relative_path).resolve()
        references_root = (skill.directory / "references").resolve()
        try:
            requested.relative_to(references_root)
        except ValueError as exc:
            raise ValueError("只允许读取当前 Skill 的 references 目录") from exc
        if requested.suffix.lower() != ".md":
            raise ValueError("只允许读取 Markdown reference")
        if not requested.is_file():
            raise FileNotFoundError(f"Reference 不存在：{relative_path}")
        return requested.read_text(encoding="utf-8")
