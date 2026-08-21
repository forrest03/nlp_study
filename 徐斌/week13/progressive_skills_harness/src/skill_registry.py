"""
Skill 注册表：扫描 skills/ 目录，解析 SKILL.md frontmatter，生成常驻索引。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class SkillMeta:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    version: str = ""
    path: Path = field(default_factory=Path)
    skill_md: Path = field(default_factory=Path)
    body: str = ""

    @property
    def index_line(self) -> str:
        """常驻层：一行摘要（Progressive Disclosure L0）"""
        desc = self.description.strip().replace("\n", " ")
        if len(desc) > 120:
            desc = desc[:117] + "..."
        trig = ", ".join(self.triggers[:5]) if self.triggers else "(see description)"
        return f"- [{self.name}]({self.name}/SKILL.md) — {desc} | triggers: {trig}"


def _parse_yaml_simple(raw: str) -> dict:
    """极简 YAML 解析（够用：标量 / 多行 >- / 列表）。"""
    data: dict = {}
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()

        if rest in (">-", ">"):
            # 折叠多行字符串
            chunks: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt and not nxt.startswith(" ") and not nxt.startswith("\t") and ":" in nxt:
                    break
                chunks.append(nxt.strip())
                i += 1
            data[key] = " ".join(c for c in chunks if c)
            continue

        if rest == "" or rest == "|":
            # 可能是列表或块
            i += 1
            items: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                m = re.match(r"^\s*-\s+(.*)$", nxt)
                if m:
                    items.append(m.group(1).strip().strip("'\""))
                    i += 1
                    continue
                if nxt and not nxt.startswith(" ") and ":" in nxt:
                    break
                i += 1
            if items:
                data[key] = items
            else:
                data[key] = ""
            continue

        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            data[key] = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
        else:
            data[key] = rest.strip("'\"")
        i += 1
    return data


def parse_skill_md(path: Path) -> SkillMeta:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        name = path.parent.name
        return SkillMeta(
            name=name,
            description=f"Skill at {path.parent.name}",
            path=path.parent,
            skill_md=path,
            body=text,
        )
    meta = _parse_yaml_simple(m.group(1))
    body = m.group(2).strip()
    name = str(meta.get("name") or path.parent.name)
    desc = str(meta.get("description") or "")
    triggers = meta.get("triggers") or []
    if isinstance(triggers, str):
        triggers = [t.strip() for t in re.split(r"[|,]", triggers) if t.strip()]
    return SkillMeta(
        name=name,
        description=desc,
        triggers=list(triggers),
        version=str(meta.get("version") or ""),
        path=path.parent,
        skill_md=path,
        body=body,
    )


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, SkillMeta] = {}
        self.reload()

    def reload(self) -> None:
        self.skills.clear()
        if not self.skills_dir.exists():
            return
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            meta = parse_skill_md(skill_md)
            self.skills[meta.name] = meta
        self.write_index()

    def write_index(self) -> Path:
        """生成常驻索引 SKILLS.md（类比课件中的 MEMORY.md 索引）。"""
        lines = [
            "# SKILLS.md — Skill 索引（常驻层 L0）",
            "",
            "本文件只保留一行摘要。完整 Skill 定义按需加载，避免 context 膨胀。",
            "",
        ]
        for meta in self.skills.values():
            lines.append(meta.index_line)
        lines.append("")
        path = self.skills_dir / "SKILLS.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def get(self, name: str) -> SkillMeta | None:
        return self.skills.get(name)

    def list_metas(self) -> list[SkillMeta]:
        return list(self.skills.values())

    def build_index_text(self) -> str:
        path = self.skills_dir / "SKILLS.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self.write_index().read_text(encoding="utf-8")

    def match_by_triggers(self, query: str) -> list[str]:
        """关键词触发初筛（零成本），供 harness 提示模型。"""
        q = query.lower()
        hits: list[tuple[int, str]] = []
        for meta in self.skills.values():
            score = 0
            for t in meta.triggers:
                if t.lower() in q:
                    score += 2
            # description 关键词弱匹配
            for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z-]{3,}", meta.description):
                if token.lower() in q and len(token) >= 2:
                    score += 1
            if meta.name.lower() in q:
                score += 3
            if score > 0:
                hits.append((score, meta.name))
        hits.sort(key=lambda x: (-x[0], x[1]))
        return [name for _, name in hits]
