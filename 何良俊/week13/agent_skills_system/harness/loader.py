"""
ProgressiveSkillLoader — 渐进式 skill 加载器
========================================================

三阶段加载策略（核心思想：尽量推迟 IO，只在需要时才读取更多内容）：

  Phase 1 — scan() 扫描整个 skills 目录，仅解析每个 SKILL.md 的 YAML frontmatter
            （name / description / version + 执行契约 params/entry/...），
            返回 SkillMeta 列表。开销极小，可一次性加载全部。

  Phase 2 — load_full(meta) 当某 skill 被匹配器选中后，才读取其 SKILL.md 完整
            正文（markdown body），返回 LoadedSkill。

  Phase 3 — 由 executor 在执行前枚举 skill 目录下的 scripts/、data/、references/
            等子目录与文件路径（见 executor._discover_scripts_inline）。

这种分阶段的目的是：当 skills 目录膨胀到几十/上百个时，启动只付 Phase 1 的
代价；运行期只对真正命中的 skill 付 Phase 2 / 3 代价。

### 声明式执行契约（frontmatter）

skill 在 SKILL.md frontmatter 中声明自己的执行契约，harness 通用执行，
**新增 skill 不需要改 harness 代码**。契约字段（全部可选）：

  params:            # 参数声明，matcher 据此让 LLM 抽参
    - name, type, description, required
  entry:             # 执行脚本相对路径（无则走"生成模式"）
  entry_input:       # data_file | stdin | args，如何把数据喂给 entry
  data_instructions: # LLM 生成数据的指令模板（支持 {param} 与 {avoid_hint} 占位）
  output_ext:        # 生成模式产物扩展名（如 .svg）
  output_subdir:     # 生成模式产物子目录（默认 skill name）
  output_name:       # 产物命名模板（默认用 topic 或首个参数）
  post_process:      # 生成模式后处理脚本（如 SVG→PNG）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# YAML frontmatter block, delimited by --- on its own line
_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(?P<fm>.*?)\n---\s*\n?(?P<body>.*)$",
    re.DOTALL,
)


@dataclass
class SkillMeta:
    """Phase 1 产物：仅 frontmatter 元数据 + 执行契约。"""

    name: str
    description: str
    version: str
    skill_dir: Path
    skill_md_path: Path
    # ---- 声明式执行契约（全部可选，组合决定执行模式）----
    params: list[dict] = field(default_factory=list)
    entry: str = ""
    entry_input: str = ""        # data_file | stdin | args
    data_instructions: str = ""
    output_ext: str = ""        # 生成模式产物扩展名
    output_subdir: str = ""      # 生成模式产物子目录
    output_name: str = ""        # 产物命名模板
    post_process: str = ""       # 生成模式后处理脚本


@dataclass
class LoadedSkill(SkillMeta):
    """Phase 2 产物：在元数据之上补充 SKILL.md 正文与目录结构发现结果。"""

    body: str = ""
    scripts: dict = field(default_factory=dict)      # name -> Path (Phase 3)
    data_files: dict = field(default_factory=dict)   # stem -> Path (Phase 3)
    references: dict = field(default_factory=dict)    # name -> Path (Phase 3)
    _full_loaded: bool = False
    _scripts_loaded: bool = False


class ProgressiveSkillLoader:
    """三阶段渐进式 skill 加载器。"""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        if not self.skills_dir.exists():
            raise FileNotFoundError(f"skills directory not found: {self.skills_dir}")
        # Phase 1 cache
        self._registry: dict[str, SkillMeta] = {}
        # Phase 2 cache
        self._full_cache: dict[str, LoadedSkill] = {}

    # ---- Phase 1: registry ---------------------------------------------
    def scan(self, *, force: bool = False) -> list[SkillMeta]:
        """扫描 skills 目录，仅解析 frontmatter。可重复调用，结果会被缓存。"""
        if self._registry and not force:
            return list(self._registry.values())

        if force:
            self._registry.clear()
            self._full_cache.clear()

        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            meta = self._parse_frontmatter(skill_md)
            if meta:
                self._registry[meta.name] = meta
        return list(self._registry.values())

    @staticmethod
    def _parse_frontmatter(skill_md: Path) -> Optional[SkillMeta]:
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            return None

        m = _FRONTMATTER_RE.match(text)
        if not m:
            return None

        fm_text = m.group("fm")
        try:
            fm = yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            # 解析失败则按空契约降级
            fm = {}

        if not isinstance(fm, dict):
            return None

        name = str(fm.get("name") or skill_md.parent.name)
        # description 可能是多行折叠字符串
        description = str(fm.get("description") or "")
        version = str(fm.get("version") or "0.0.0")

        params = fm.get("params") or []
        if not isinstance(params, list):
            params = []

        return SkillMeta(
            name=name,
            description=description,
            version=version,
            skill_dir=skill_md.parent,
            skill_md_path=skill_md,
            params=[dict(p) for p in params if isinstance(p, dict)],
            entry=str(fm.get("entry") or ""),
            entry_input=str(fm.get("entry_input") or ""),
            data_instructions=str(fm.get("data_instructions") or ""),
            output_ext=str(fm.get("output_ext") or ""),
            output_subdir=str(fm.get("output_subdir") or ""),
            output_name=str(fm.get("output_name") or ""),
            post_process=str(fm.get("post_process") or ""),
        )

    # ---- Phase 2: full markdown body -----------------------------------
    def load_full(self, meta: SkillMeta) -> LoadedSkill:
        """读取 SKILL.md 完整正文。命中缓存直接返回。"""
        if meta.name in self._full_cache:
            return self._full_cache[meta.name]

        try:
            text = meta.skill_md_path.read_text(encoding="utf-8")
        except OSError as e:
            raise RuntimeError(f"failed to read {meta.skill_md_path}: {e}") from e

        m = _FRONTMATTER_RE.match(text)
        body = m.group("body") if m else text

        loaded = LoadedSkill(
            name=meta.name,
            description=meta.description,
            version=meta.version,
            skill_dir=meta.skill_dir,
            skill_md_path=meta.skill_md_path,
            params=meta.params,
            entry=meta.entry,
            entry_input=meta.entry_input,
            data_instructions=meta.data_instructions,
            output_ext=meta.output_ext,
            output_subdir=meta.output_subdir,
            output_name=meta.output_name,
            post_process=meta.post_process,
            body=body,
            _full_loaded=True,
        )
        self._full_cache[meta.name] = loaded
        return loaded

    def get(self, name: str) -> Optional[SkillMeta]:
        if not self._registry:
            self.scan()
        return self._registry.get(name)
