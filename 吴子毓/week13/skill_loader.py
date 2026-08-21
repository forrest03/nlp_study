"""
skill_loader.py — Harness 数据层
================================
职责：扫描 skills/ 目录、解析 SKILL.md、维护注册表、按需渐进式加载、mtime 热重载。

对标原项目：
  - agent_memory_system/src/memory_loader.py   → 分层渐进加载 + MemoryLayer/SystemPromptResult 数据结构
  - agent_memory_system/src/heartbeat_parser.py → Markdown 块解析（标记定位 + split + key:value）
  - agent_memory_system/src/scheduler.py        → _check_reload 的 mtime 热重载机制

设计原则：
  本文件只做"数据"——不调 LLM、不执行脚本、不打印进度。
  所有 LLM 调用与执行都在 agent.py，保持数据层纯净可测试。
"""
# -*- coding: utf-8 -*-
from __future__ import annotations  # 让 Path | None 等类型注解延迟求值，兼容 Python 3.8

# ════════════════════════════════════════════════════════════════════════════
# 模块 1：导入与路径设置
# ════════════════════════════════════════════════════════════════════════════
# - import re / yaml（或手写 front-matter 解析，避免引入 pyyaml 依赖）
# - from dataclasses import dataclass, field
# - from pathlib import Path
# - from datetime import datetime
# - SKILLS_DIR 常量：Path(__file__).parent.parent / "skills"
#   （skills 目录与 homework 同级，即项目根下的 skills/）
#
# 关键细节：
#   - 不依赖 agent_memory_system，保持 harness 自包含；
#   - 若需 LLM 调用，由 agent.py 注入，本文件不直接 import llm_config。

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Windows OpenMP 兼容（与原项目 agent.py 一致，即便本文件不用 faiss 也无害）
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# skills/ 目录：homework/ 的父目录下
SKILLS_DIR = Path(__file__).parent.parent / "skills"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ════════════════════════════════════════════════════════════════════════════
# 模块 2：数据结构定义（对标 MemoryLayer / SystemPromptResult）
# ════════════════════════════════════════════════════════════════════════════
#
# @dataclass SkillMeta
#   —— 轻量元数据，仅来自 front-matter，启动时全量加载（渐进式第 1 层）
#   字段：
#     name: str               # front-matter 的 name
#     description: str        # front-matter 的 description（用于两阶段匹配的 Stage 1）
#     version: str            # front-matter 的 version
#     base_dir: Path          # SKILL.md 所在目录（用于后续拼 scripts/、references/ 路径）
#     skill_md_path: Path     # SKILL.md 完整路径
#     mtime: float            # SKILL.md 的 stat().st_mtime，热重载比对用
#   方法：无（纯数据）
#
# @dataclass LoadedSkill
#   —— 完整加载结果，包含 body / references（渐进式第 2、3 层）
#   字段：
#     meta: SkillMeta
#     body: str = ""              # SKILL.md 正文（去掉 front-matter 后的 Markdown）
#     body_sections: dict = {}    # body 按 ## 标题切分后的 {标题: 内容}，便于按段取用
#     references: dict = {}       # {ref文件名: 内容}，来自 base_dir/references/*.md
#     script_path: Path | None    # base_dir/scripts/ 下主脚本路径（自动探测 .py/.ts）
#     char_count: int = 0         # __post_init__ 里算 body + references 总字符数
#   方法：__post_init__ 计算 char_count
#
# 关键细节：
#   - 分两层 dataclass 的原因：SkillMeta 廉价（启动全量），LoadedSkill 重（按需构造），
#     对标 memory_loader 里 MemoryLayer（单层）vs SystemPromptResult（聚合）的分层思想；
#   - body_sections 用 dict 而非 list，便于 execute_skill 时按段名（如"执行流程""输出规则"）取用。


@dataclass
class SkillMeta:
    """轻量元数据——渐进式第 1 层，启动时全量加载。"""
    name: str
    description: str
    version: str
    base_dir: Path
    skill_md_path: Path
    mtime: float


@dataclass
class LoadedSkill:
    """完整加载结果——渐进式第 2、3 层，按需构造。"""
    meta: SkillMeta
    body: str = ""
    body_sections: dict = field(default_factory=dict)   # 用 field 避免 mutable default 警告
    references: dict = field(default_factory=dict)
    script_path: Path | None = None
    char_count: int = 0

    def __post_init__(self):
        # 构造完成时自动算字符数（references 此时通常为空，load_references 后会重算）
        self.char_count = len(self.body) + sum(len(v) for v in self.references.values())


# ════════════════════════════════════════════════════════════════════════════
# 模块 3：SKILL.md 解析工具函数（对标 heartbeat_parser._parse_task_block）
# ════════════════════════════════════════════════════════════════════════════
#
# def _parse_front_matter(text: str) -> tuple[dict, str]
#   —— 拆分 SKILL.md 的 YAML front-matter 与 Markdown 正文
#   关键细节：
#     - 文件首行应为 "---"，找第二个 "---" 作结束标记；
#     - front-matter 段用简单 key: value 解析（不引 pyyaml）：
#         * name / version：直接 strip
#         * description：支持单行或 YAML 多行 ">-" 块（参考 flash-card SKILL.md 的多行 description）
#           → 检测到 ">-" 后，取其后到缩进回退的所有行拼接
#     - 返回 (meta_dict, body_text)；body_text 为去掉 front-matter 后的纯 Markdown
#     - 容错：首行无 "---" 时，meta_dict={}，body_text=原文（降级为无元数据 skill）
#
# def _split_body_sections(body: str) -> dict[str, str]
#   —— 把正文按 "## " 二级标题切成 {标题: 内容}
#   关键细节：
#     - re.split(r"(?=^## )", body, flags=re.MULTILINE) 切块；
#     - 每块首行 "## 标题" 提取为 key，其余为 value；
#     - 对标 heartbeat_parser 用 re.split(r"(?=### TASK:)") 切任务块的同款手法；
#     - 用于后续按需取"触发场景""执行流程""输出规则"等段，避免每次解析全量。
#
# def _detect_script(base_dir: Path) -> Path | None
#   —— 自动探测 base_dir/scripts/ 下的主脚本
#   关键细节：
#     - 优先级：main.py > main.ts > make_*.py > 目录内唯一脚本；
#     - 不存在返回 None（纯文本 skill 也合法）；
#     - 参考 baoyu-diagram/scripts/main.ts 与 flash-card/scripts/make_flashcard.py 的命名习惯。


# YAML 多行块标记：>- / > / |- / | 都表示后续缩进行是一个值
_MULTILINE_MARKERS = (">-", ">", "|-", "|")


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """拆分 SKILL.md 的 YAML front-matter 与 Markdown 正文，返回 (meta_dict, body)。"""
    lines = text.splitlines()
    # 容错：首行不是 "---" → 无 front-matter，原文整体当 body
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    # 找闭合的 "---"
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text.strip()

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:]).strip()

    meta: dict = {}
    i = 0
    while i < len(fm_lines):
        line = fm_lines[i]
        if ":" not in line:
            i += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if val in _MULTILINE_MARKERS:
            # 多行值：收集后续缩进行（以空格/tab开头），到缩进回退为止
            collected = []
            i += 1
            while i < len(fm_lines) and (fm_lines[i].startswith(("  ", "\t"))):
                collected.append(fm_lines[i].strip())
                i += 1
            meta[key] = " ".join(c for c in collected if c)
        else:
            meta[key] = val
            i += 1

    return meta, body


def _split_body_sections(body: str) -> dict:
    """把正文按 "## " 二级标题切成 {标题: 内容}。"""
    if not body:
        return {}
    # 正向前瞻切分：每个 "## " 起始处切块，保留块首的 "## "
    blocks = re.split(r"(?=^## )", body, flags=re.MULTILINE)
    sections: dict = {}
    for block in blocks:
        block = block.strip()
        if not block.startswith("## "):
            continue
        nl = block.find("\n")
        if nl == -1:
            title = block[3:].strip()
            content = ""
        else:
            title = block[3:nl].strip()
            content = block[nl + 1:].strip()
        sections[title] = content
    return sections


def _detect_script(base_dir: Path) -> Path | None:
    """自动探测 base_dir/scripts/ 下的主脚本，按优先级返回。"""
    scripts_dir = base_dir / "scripts"
    if not scripts_dir.exists():
        return None

    # 优先级 1：main.py
    main_py = scripts_dir / "main.py"
    if main_py.exists():
        return main_py
    # 优先级 2：main.ts
    main_ts = scripts_dir / "main.ts"
    if main_ts.exists():
        return main_ts
    # 优先级 3：make_*.py（如 flash-card 的 make_flashcard.py）
    make_scripts = sorted(scripts_dir.glob("make_*.py"))
    if make_scripts:
        return make_scripts[0]
    # 优先级 4：目录内唯一脚本文件
    candidates = [
        p for p in scripts_dir.iterdir()
        if p.is_file() and p.suffix in (".py", ".ts", ".js")
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


# ════════════════════════════════════════════════════════════════════════════
# 模块 4：SkillRegistry 类（核心，对标 scheduler 的 _load_tasks + _check_reload）
# ════════════════════════════════════════════════════════════════════════════
#
# class SkillRegistry:
#     def __init__(self, skills_dir: Path = SKILLS_DIR):
#         self.skills_dir = skills_dir
#         self._metas: dict[str, SkillMeta] = {}     # name -> SkillMeta（注册表，启动全量）
#         self._loaded: dict[str, LoadedSkill] = {}  # name -> LoadedSkill（按需缓存，命中才填）
#         self._dir_mtime: float = 0                  # skills/ 目录 mtime，热重载粗筛用


class SkillRegistry:
    """skill 注册表 + 渐进式加载器 + 热重载。"""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir
        self._metas: dict[str, SkillMeta] = {}
        self._loaded: dict[str, LoadedSkill] = {}
        self._dir_mtime: float = 0

    # ── 4.1 扫描注册（启动调用一次 + 热重载时调用）──────────────────────────
    # def scan_skills(self) -> list[SkillMeta]:
    #     """
    #     遍历 skills/*/SKILL.md，解析 front-matter，填充 self._metas。
    #     返回注册成功的 SkillMeta 列表。
    #     """
    #     关键细节：
    #       - 用 skills_dir.glob("*/SKILL.md") 找所有 skill（不递归，一层）；
    #       - 对每个 SKILL.md：read_text → _parse_front_matter → 构造 SkillMeta
    #         （mtime = skill_md_path.stat().st_mtime）；
    #       - name 冲突时后者覆盖前者并 log warning（不抛异常，保证启动不中断）；
    #       - 解析失败的 skill 跳过并记录，不影响其他 skill 注册；
    #       - self._dir_mtime = skills_dir.stat().st_mtime（粗筛快照）；
    #       - 对标 scheduler._load_tasks() 清旧 job 重注册的语义——这里清空 _metas 再填。

    def scan_skills(self) -> list[SkillMeta]:
        """遍历 skills/*/SKILL.md，解析 front-matter，填充 self._metas。"""
        self._metas.clear()

        if not self.skills_dir.exists():
            logger.error(f"skills 目录不存在：{self.skills_dir}")
            self._dir_mtime = 0
            return []

        for skill_md in self.skills_dir.glob("*/SKILL.md"):
            try:
                text = skill_md.read_text(encoding="utf-8")
                meta_dict, _ = _parse_front_matter(text)
                name = meta_dict.get("name") or skill_md.parent.name
                meta = SkillMeta(
                    name=name,
                    description=meta_dict.get("description", ""),
                    version=meta_dict.get("version", ""),
                    base_dir=skill_md.parent,
                    skill_md_path=skill_md,
                    mtime=skill_md.stat().st_mtime,
                )
                if name in self._metas:
                    logger.warning(f"skill name 冲突，后者覆盖前者：{name}")
                self._metas[name] = meta
            except Exception as e:
                logger.error(f"解析 {skill_md} 失败：{e}")

        self._dir_mtime = self.skills_dir.stat().st_mtime
        logger.info(f"已扫描 {len(self._metas)} 个 skill")
        return list(self._metas.values())

    # ── 4.2 渐进式加载第 1 层：front-matter（启动全量）──────────────────────
    # def load_all_meta(self) -> dict[str, SkillMeta]:
    #     """
    #     启动时调一次：若 _metas 为空则 scan_skills()，返回 _metas 副本。
    #     这是最轻的加载层，所有 skill 的 description 都在内存里供 Stage 1 匹配。
    #     """
    #     关键细节：
    #       - 惰性：_metas 已填充则直接返回，避免重复扫描；
    #       - 返回副本防止外部修改内部注册表。

    def load_all_meta(self) -> dict[str, SkillMeta]:
        """启动时调一次，返回 name -> SkillMeta 的副本（渐进式第 1 层）。"""
        if not self._metas:
            self.scan_skills()
        return dict(self._metas)

    # ── 4.3 渐进式加载第 2 层：body（命中才加载）────────────────────────────
    # def load_body(self, name: str) -> LoadedSkill:
    #     """
    #     Stage 1/2 匹配命中某 skill 后调用：读取其正文，构造 LoadedSkill（不含 references）。
    #     结果缓存到 _loaded，重复触发同 skill 不再读磁盘。
    #     """
    #     关键细节：
    #       - 查 _metas[name] 拿 SkillMeta，缺失则 raise KeyError；
    #       - 若 _loaded[name] 已存在且其 meta.mtime == 当前文件 mtime → 直接返回缓存；
    #       - 否则 read_text → _parse_front_matter 取 body → _split_body_sections →
    #         _detect_script → 构造 LoadedSkill（references 暂留空 dict）→ 存 _loaded；
    #       - 对标 memory_loader._extract_memory_entries() 的"按需切片"思想：
    #         只在真正需要时读正文，而不是启动全量读。

    def load_body(self, name: str) -> LoadedSkill:
        """命中某 skill 后调：读正文构造 LoadedSkill（渐进式第 2 层），结果缓存。"""
        if name not in self._metas:
            raise KeyError(f"未注册的 skill：{name}")

        meta = self._metas[name]
        current_mtime = meta.skill_md_path.stat().st_mtime

        # 缓存命中：文件未改动则直接返回
        cached = self._loaded.get(name)
        if cached is not None and cached.meta.mtime == current_mtime:
            return cached

        text = meta.skill_md_path.read_text(encoding="utf-8")
        _, body = _parse_front_matter(text)
        meta.mtime = current_mtime  # 顺带刷新 mtime
        loaded = LoadedSkill(
            meta=meta,
            body=body,
            body_sections=_split_body_sections(body),
            references={},  # 第 3 层留空，load_references 时才填
            script_path=_detect_script(meta.base_dir),
        )
        self._loaded[name] = loaded
        return loaded

    # ── 4.4 渐进式加载第 3 层：references（执行前才加载，最重）──────────────
    # def load_references(self, name: str) -> dict[str, str]:
    #     """
    #     execute_skill 执行前调用：读取 base_dir/references/*.md 全部内容。
    #     返回 {文件名: 正文}，并写回 _loaded[name].references。
    #     """
    #     关键细节：
    #       - 先确保 load_body(name) 已执行（_loaded[name] 存在）；
    #       - 用 meta.base_dir.glob("references/*.md") 找参考文件；
    #       - 逐个 read_text，按 stem（去 .md）作 key；
    #       - references 可能为空（如 flash-card 无 references），返回 {}；
    #       - 这是最重的层——参考 baoyu-diagram/references/ 下有 4 个 md，全量读成本高，
    #         所以放到最后，仅当确定要执行该 skill 时才读。

    def load_references(self, name: str) -> dict:
        """执行前调：读 base_dir/references/*.md（渐进式第 3 层，最重），写回缓存。"""
        if name not in self._loaded:
            self.load_body(name)  # 确保 body 已加载
        loaded = self._loaded[name]

        # 已加载过则直接返回（避免重复读盘）
        if loaded.references:
            return loaded.references

        refs_dir = loaded.meta.base_dir / "references"
        if not refs_dir.exists():
            loaded.references = {}
            return {}

        refs: dict = {}
        for ref_file in sorted(refs_dir.glob("*.md")):
            try:
                refs[ref_file.stem] = ref_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"读取参考文件失败 {ref_file}：{e}")
        loaded.references = refs
        # references 填充后重算字符数
        loaded.char_count = len(loaded.body) + sum(len(v) for v in refs.values())
        return refs

    # ── 4.5 mtime 热重载（对标 scheduler._check_reload）─────────────────────
    # def reload_if_changed(self) -> bool:
    #     """
    #     主循环每轮调用：检测 skills/ 目录 mtime 是否变化，变化则重扫注册表。
    #     返回是否发生了重载。
    #     """
    #     关键细节：
    #       - 先比 self._dir_mtime（目录级粗筛，O(1)）；
    #       - 目录 mtime 变了 → 再逐个比 _metas 里各 skill_md_path 的 mtime，
    #         只重扫变化的 skill（细粒度，避免全量重解析）；
    #       - 重扫后清掉 _loaded 中对应缓存（body/references 失效需重读）；
    #       - 新增的 skill 目录会被 glob 带出（因为重新 scan）；
    #       - 删除的 skill 从 _metas 移除；
    #       - 对标 scheduler._check_reload() 每 60s 比 HEARTBEAT_PATH.stat().st_mtime 的手法，
    #         但这里由 agent.py 主循环每轮触发，不需独立定时器。

    def reload_if_changed(self) -> bool:
        """
        主循环每轮调用：比对 skills/ 目录 mtime，变化则重扫注册表并失效对应缓存。
        返回是否发生了重载。

        注意：目录 mtime 只反映直接子项的增删，编辑已有 SKILL.md 正文不会触发目录 mtime 变化。
        那种情况由 agent.py 的 /reload 命令强制重扫兜底。
        """
        if not self.skills_dir.exists():
            return False
        try:
            dir_mtime = self.skills_dir.stat().st_mtime
        except OSError:
            return False

        # 第 1 级粗筛：目录 mtime 没变就直接返回
        if dir_mtime == self._dir_mtime:
            return False

        # 目录变了 → 保存旧 meta 快照后重扫
        old_metas = dict(self._metas)
        self.scan_skills()  # 会刷新 self._metas 与 self._dir_mtime(所以不需要new_metas - old_metas)

        old_names = set(old_metas.keys())
        new_names = set(self._metas.keys())

        # 删除的 skill：清缓存
        for name in old_names - new_names:
            self._loaded.pop(name, None)

        # 改动过的 skill（mtime 不同）：清缓存，强制下次 load_body 重读
        for name in new_names & old_names:
            if self._metas[name].mtime != old_metas[name].mtime:
                self._loaded.pop(name, None)

        logger.info(f"热重载完成，当前 {len(self._metas)} 个 skill")
        return True

    # ── 4.6 查询接口 ────────────────────────────────────────────────────────
    # def list_skills(self) -> list[SkillMeta]:
    #     """返回所有已注册 skill 的 meta 列表（按 name 排序）。"""
    #
    # def get_meta(self, name: str) -> SkillMeta | None:
    #     """按 name 取单个 SkillMeta，不存在返回 None。"""
    #
    # def get_loaded(self, name: str) -> LoadedSkill | None:
    #     """按 name 取已加载的 LoadedSkill（未 load_body 过则返回 None）。"""

    def list_skills(self) -> list[SkillMeta]:
        """返回所有已注册 skill 的 meta 列表（按 name 排序）。"""
        return sorted(self._metas.values(), key=lambda m: m.name)

    def get_meta(self, name: str) -> SkillMeta | None:
        """按 name 取单个 SkillMeta，不存在返回 None。"""
        return self._metas.get(name)

    def get_loaded(self, name: str) -> LoadedSkill | None:
        """按 name 取已加载的 LoadedSkill（未 load_body 过则返回 None）。"""
        return self._loaded.get(name)


# ════════════════════════════════════════════════════════════════════════════
# 模块 5：模块级便捷函数（供 agent.py 直接 import 调用）
# ════════════════════════════════════════════════════════════════════════════
#
# def build_registry(skills_dir: Path = SKILLS_DIR) -> SkillRegistry:
#     """工厂函数：构造 SkillRegistry 并立即 scan_skills()，返回就绪的注册表。"""
#     关键细节：
#       - agent.py 启动时 `from skill_loader import build_registry` 一行拿到就绪注册表；
#       - scan 失败不抛异常，返回空注册表（log error），保证 harness 不崩。


def build_registry(skills_dir: Path = SKILLS_DIR) -> SkillRegistry:
    """工厂函数：构造 SkillRegistry 并立即 scan_skills()，返回就绪的注册表。"""
    registry = SkillRegistry(skills_dir)
    try:
        registry.scan_skills()
    except Exception as e:
        logger.error(f"build_registry 扫描失败：{e}")
    return registry


# ════════════════════════════════════════════════════════════════════════════
# 设计说明（不写代码，仅备忘）
# ════════════════════════════════════════════════════════════════════════════
# 渐进式三层加载总结：
#   第 1 层  front-matter  →  load_all_meta()    启动全量，廉价，供 Stage 1 匹配
#   第 2 层  body          →  load_body(name)    命中才读，中等，供 Stage 2 / 执行流程解析
#   第 3 层  references    →  load_references()  执行前读，最重，供 execute_skill 注入上下文
#
# 与原项目 memory_loader 的对应：
#   SOUL.md（全量）     ↔ front-matter（全量）
#   USER.md（全量）     ↔ body（按需）
#   MEMORY 近 N 条（切片）↔ references（按需）
#   分层 join + char_count ↔ LoadedSkill.char_count


# ── 自测入口：直接 python skill_loader.py 可验证扫描与加载 ──────────────────
if __name__ == "__main__":
    reg = build_registry()
    print(f"\n已注册 {len(reg.list_skills())} 个 skill：")
    for m in reg.list_skills():
        print(f"  - {m.name}  v{m.version}  [{m.skill_md_path.parent.name}]")
        print(f"    desc: {m.description[:60]}...")

    # 演示渐进式加载第 2、3 层（取第一个 skill）
    if reg.list_skills():
        first = reg.list_skills()[0]
        print(f"\n--- 渐进加载 {first.name} ---")
        loaded = reg.load_body(first.name)
        print(f"body 字符数: {len(loaded.body)}")
        print(f"body 段落: {list(loaded.body_sections.keys())}")
        print(f"script: {loaded.script_path}")
        refs = reg.load_references(first.name)
        print(f"references: {list(refs.keys()) if refs else '（无）'}")
        print(f"总字符数: {loaded.char_count}")
