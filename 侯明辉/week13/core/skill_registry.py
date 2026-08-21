"""
skill_registry.py — Skill 注册表

职责：
  1. 启动时扫描 skills/MEMORY.md，产出 SkillMeta 列表（常驻轻量索引）
  2. 按需加载单个 SKILL.md 的完整内容（Frontmatter + Body）
  3. 提供触发词匹配（关键词正则）

设计原则：
  - MEMORY.md 索引只保存「名字 + 一行描述」，常驻 < 200 tokens
  - 完整 SKILL.md 只在匹配触发后才加载到 Context
  - YAML Frontmatter 解析失败时给出清晰错误
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── 数据结构 ───────────────────────────────────────────────────────────────

@dataclass
class SkillMeta:
    """MEMORY.md 索引行的轻量表示（始终驻留 Context）"""
    name: str            # 唯一标识
    path: str            # 相对路径，如 "weather/SKILL.md"
    description: str     # 一行描述
    trigger: str = ""    # Frontmatter 触发词（懒加载时才有）
    tools: list[str] = field(default_factory=list)


@dataclass
class SkillFull:
    """单个 SKILL.md 的完整内容（按需加载）"""
    meta: dict           # Frontmatter 解析结果
    body: str            # 去掉 frontmatter 后的 markdown 正文
    source_path: Path

    @property
    def name(self) -> str:
        return self.meta.get("name", "")

    @property
    def trigger_patterns(self) -> list[str]:
        """从 trigger 字段拆出多个关键词（用 '|' 或逗号分隔）"""
        raw = self.meta.get("trigger", "")
        if not raw:
            return []
        parts = re.split(r"[|,]", str(raw))
        return [p.strip() for p in parts if p.strip()]


# ── SKILL.md 解析 ──────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_skill_md(path: Path) -> SkillFull:
    """解析单个 SKILL.md：剥离 Frontmatter + 保留正文"""
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"SKILL.md 缺少 Frontmatter: {path}")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"SKILL.md Frontmatter YAML 解析失败 {path}: {e}")
    if not isinstance(meta, dict):
        raise ValueError(f"SKILL.md Frontmatter 必须是 mapping: {path}")
    body = text[m.end():].strip()
    return SkillFull(meta=meta, body=body, source_path=path)


# ── MEMORY.md 索引 ─────────────────────────────────────────────────────────

_INDEX_LINE_RE = re.compile(r"-\s*\[(?P<name>[^\]]+)\]\((?P<path>[^)]+)\)\s*[—\-:]\s*(?P<desc>.+)")


def parse_memory_index(memory_path: Path) -> list[SkillMeta]:
    """从 MEMORY.md 解析索引行（一行一个 Skill）"""
    if not memory_path.exists():
        return []
    text = memory_path.read_text(encoding="utf-8")
    out: list[SkillMeta] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _INDEX_LINE_RE.match(line)
        if not m:
            continue
        out.append(SkillMeta(
            name=m.group("name").strip(),
            path=m.group("path").strip(),
            description=m.group("desc").strip(),
        ))
    return out


def render_memory_index(metas: list[SkillMeta]) -> str:
    """把索引列表渲染回 MEMORY.md（CLI /skills 用）"""
    lines = ["# Skills 索引（MEMORY.md）",
             "",
             "> 启动时由 SkillRegistry 自动加载，常驻 < 200 tokens。",
             "> 触发匹配后才会加载完整 SKILL.md 正文。",
             ""]
    for m in metas:
        lines.append(f"- [{m.name}]({m.path}) — {m.description}")
    return "\n".join(lines) + "\n"


# ── 注册表 ────────────────────────────────────────────────────────────────

class SkillRegistry:
    """Skills 目录的入口：索引 + 单个 Skill 加载"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self._memory_path = self.skills_dir / "MEMORY.md"
        self._index: list[SkillMeta] = parse_memory_index(self._memory_path)
        self._cache: dict[str, SkillFull] = {}
        # 主动读取每个 SKILL.md 的 Frontmatter，填充 trigger/tools
        # （轻量级：只解析 YAML，不读 body）
        self._hydrate_metadata()

    def _hydrate_metadata(self) -> None:
        """从每个 SKILL.md 的 Frontmatter 提取 trigger/tools 到索引"""
        for meta in self._index:
            path = self.skills_dir / meta.path
            if not path.exists():
                continue
            try:
                skill = parse_skill_md(path)
                meta.trigger = skill.meta.get("trigger", "") or ""
                meta.tools = skill.meta.get("tools", []) or []
            except (ValueError, OSError):
                # 单个失败不影响其他
                continue

    # ── 索引（常驻）─────────────────────────────────────────────────────
    def list_all(self) -> list[SkillMeta]:
        return list(self._index)

    def get_index_summary(self) -> str:
        """一行字符串，可直接拼到 system prompt（< 200 tokens）"""
        if not self._index:
            return "（暂无已注册 Skill）"
        lines = ["可用 Skills（仅索引，按需加载完整定义）："]
        for m in self._index:
            tools_hint = ""
            if m.tools:
                tools_hint = f" ｜ 工具: {', '.join(m.tools)}"
            lines.append(f"  - {m.name}: {m.description}{tools_hint}")
        return "\n".join(lines)

    def find_by_name(self, name: str) -> SkillMeta | None:
        return next((m for m in self._index if m.name == name), None)

    # ── 完整加载（按需）────────────────────────────────────────────────
    def load_skill(self, name: str) -> SkillFull:
        """加载 SKILL.md 完整内容（带缓存）"""
        if name in self._cache:
            return self._cache[name]
        meta = self.find_by_name(name)
        if meta is None:
            raise KeyError(f"Skill 不存在: {name}")
        path = self.skills_dir / meta.path
        skill = parse_skill_md(path)
        # 回填 Frontmatter 里的 trigger/tools 到索引（方便后续匹配）
        meta.trigger = skill.meta.get("trigger", "")
        meta.tools = skill.meta.get("tools", []) or []
        self._cache[name] = skill
        return skill

    def load_multiple(self, names: list[str]) -> list[SkillFull]:
        out: list[SkillFull] = []
        for n in names:
            try:
                out.append(self.load_skill(n))
            except (KeyError, ValueError):
                # 单个失败不中断整体加载
                continue
        return out
