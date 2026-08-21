"""
skill_executor.py — Skill 上下文构建

职责：
  1. 把 SkillRegistry 的索引渲染成可注入 system prompt 的摘要
  2. 把已激活的 Skill 完整 body 渲染成可注入的块
  3. 单一入口 build_skill_context() 用于 Runner 拼接

设计要点：
  - 索引永远注入（< 200 tokens）
  - 已激活 Skill 完整内容追加（每次加载进入 context）
  - 同一 Skill 同一会话内激活一次，避免重复注入
"""
from __future__ import annotations

from core.skill_registry import SkillRegistry, SkillFull


def render_skill_block(skill: SkillFull) -> str:
    """把单个 Skill 完整内容渲染成一块 markdown"""
    name = skill.meta.get("name", "<unnamed>")
    desc = skill.meta.get("description", "")
    tools = skill.meta.get("tools") or []
    tools_line = f"\n**可用工具**：{', '.join(tools)}" if tools else ""
    desc_line = f"\n> {desc}" if desc else ""
    return (
        f"\n## Active Skill: {name}{desc_line}{tools_line}\n\n"
        f"{skill.body}\n"
    )


def build_skill_context(
    registry: SkillRegistry,
    active_skills: list[SkillFull],
) -> str:
    """组装 system prompt 的 Skill 段：索引 + 已激活 Skill 详情"""
    parts: list[str] = []
    # 1. 索引（常驻）
    index_summary = registry.get_index_summary()
    parts.append(index_summary)
    # 2. 已激活 Skill 详情
    if active_skills:
        parts.append("\n---\n当前已激活的 Skill（完整定义）：")
        for skill in active_skills:
            parts.append(render_skill_block(skill))
    return "\n".join(parts)
