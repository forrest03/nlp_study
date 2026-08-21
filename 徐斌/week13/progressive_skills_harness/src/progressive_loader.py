"""
渐进式披露加载器（Progressive Disclosure）

层级：
  L0 常驻层 — Skill 索引（SKILLS.md），每次请求都注入
  L1 触发层 — 完整 SKILL.md，activate_skill 后注入
  L2 执行层 — references / data 等资源，read_skill_file 按需注入
  释放     — release_skill 后从活跃集合移除（后续轮次不再注入全文）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from skill_registry import SkillMeta, SkillRegistry


def estimate_tokens(text: str) -> int:
    """粗估 token：中文≈字数，英文≈词数*1.3，混合取 max(字/1.5, 词*1.3)。"""
    if not text:
        return 0
    chars = len(text)
    words = len(text.split())
    return max(int(chars / 1.5), int(words * 1.3), 1)


@dataclass
class LoadEvent:
    level: str  # L0 | L1 | L2 | release
    skill: str
    path: str
    tokens: int
    detail: str = ""


@dataclass
class ProgressiveLoader:
    registry: SkillRegistry
    project_root: Path
    active_skills: dict[str, str] = field(default_factory=dict)  # name -> full md text
    loaded_resources: dict[str, str] = field(default_factory=dict)  # "skill:rel" -> content
    events: list[LoadEvent] = field(default_factory=list)

    def index_text(self) -> str:
        text = self.registry.build_index_text()
        self.events.append(
            LoadEvent(
                level="L0",
                skill="*",
                path="skills/SKILLS.md",
                tokens=estimate_tokens(text),
                detail="常驻索引",
            )
        )
        return text

    def activate(self, name: str) -> dict:
        meta = self.registry.get(name)
        if meta is None:
            return {"ok": False, "error": f"unknown skill: {name}"}
        full = meta.skill_md.read_text(encoding="utf-8")
        # 替换 {baseDir} 为真实路径，方便模型执行脚本
        base = str(meta.path.resolve())
        full_resolved = full.replace("{baseDir}", base)
        self.active_skills[name] = full_resolved
        tokens = estimate_tokens(full_resolved)
        self.events.append(
            LoadEvent(
                level="L1",
                skill=name,
                path=str(meta.skill_md.relative_to(self.project_root)),
                tokens=tokens,
                detail="加载完整 Skill 定义",
            )
        )
        return {
            "ok": True,
            "skill": name,
            "level": "L1",
            "tokens": tokens,
            "base_dir": base,
            "content": full_resolved,
        }

    def read_resource(self, skill: str, relative_path: str) -> dict:
        meta = self.registry.get(skill)
        if meta is None:
            return {"ok": False, "error": f"unknown skill: {skill}"}
        if skill not in self.active_skills:
            return {
                "ok": False,
                "error": f"skill '{skill}' 尚未 activate，请先 activate_skill",
            }

        rel = relative_path.lstrip("/").replace("\\", "/")
        if ".." in rel.split("/"):
            return {"ok": False, "error": "path traversal denied"}

        target = (meta.path / rel).resolve()
        try:
            target.relative_to(meta.path.resolve())
        except ValueError:
            return {"ok": False, "error": "resource must stay inside skill directory"}

        if not target.exists() or not target.is_file():
            return {"ok": False, "error": f"file not found: {rel}"}

        content = target.read_text(encoding="utf-8")
        key = f"{skill}:{rel}"
        self.loaded_resources[key] = content
        tokens = estimate_tokens(content)
        self.events.append(
            LoadEvent(
                level="L2",
                skill=skill,
                path=str(target.relative_to(self.project_root)),
                tokens=tokens,
                detail=f"按需加载资源 {rel}",
            )
        )
        return {
            "ok": True,
            "skill": skill,
            "level": "L2",
            "path": rel,
            "tokens": tokens,
            "content": content,
        }

    def release(self, name: str | None = None) -> dict:
        if name is None:
            released = list(self.active_skills.keys())
            self.active_skills.clear()
            # 只清该 skill 相关 resource
            self.loaded_resources.clear()
        else:
            released = [name] if name in self.active_skills else []
            self.active_skills.pop(name, None)
            self.loaded_resources = {
                k: v for k, v in self.loaded_resources.items() if not k.startswith(f"{name}:")
            }
        for s in released:
            self.events.append(
                LoadEvent(
                    level="release",
                    skill=s,
                    path="",
                    tokens=0,
                    detail="任务完成，释放 Skill 全文",
                )
            )
        return {"ok": True, "released": released}

    def snapshot(self) -> dict:
        index = self.registry.build_index_text()
        l0 = estimate_tokens(index)
        l1 = sum(estimate_tokens(v) for v in self.active_skills.values())
        l2 = sum(estimate_tokens(v) for v in self.loaded_resources.values())
        # 全量加载对比：所有 SKILL.md + 所有 references
        full_load = 0
        for meta in self.registry.list_metas():
            full_load += estimate_tokens(meta.skill_md.read_text(encoding="utf-8"))
            for p in meta.path.rglob("*"):
                if p.is_file() and p.suffix in {".md", ".json", ".py", ".ts"} and p.name != "SKILL.md":
                    if "node_modules" in p.parts:
                        continue
                    try:
                        full_load += estimate_tokens(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
        current = l0 + l1 + l2
        return {
            "l0_tokens": l0,
            "l1_tokens": l1,
            "l2_tokens": l2,
            "current_tokens": current,
            "full_load_tokens": full_load,
            "saved_tokens": max(full_load - current, 0),
            "active_skills": list(self.active_skills.keys()),
            "loaded_resources": list(self.loaded_resources.keys()),
            "events": [
                {
                    "level": e.level,
                    "skill": e.skill,
                    "path": e.path,
                    "tokens": e.tokens,
                    "detail": e.detail,
                }
                for e in self.events
            ],
        }

    def context_blocks(self) -> list[str]:
        """组装当前应注入的 Skill 相关 context 块。"""
        blocks: list[str] = []
        if self.active_skills:
            blocks.append("## Active Skills (L1)\n")
            for name, content in self.active_skills.items():
                blocks.append(f"### Skill: {name}\n\n{content}\n")
        if self.loaded_resources:
            blocks.append("## Skill Resources (L2)\n")
            for key, content in self.loaded_resources.items():
                blocks.append(f"### Resource: {key}\n\n{content}\n")
        return blocks
