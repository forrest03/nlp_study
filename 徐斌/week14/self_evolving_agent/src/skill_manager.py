"""
Skill 管理器：读取 / 创建 / 局部 patch，并持久化版本历史。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


class SkillManager:
    """
    三重持久化：
      - skills/{name}/SKILL.md
      - outputs/skill_versions/{name}_history.json
      - outputs/skill_snapshots/{name}_v{N}.md
    """

    def __init__(self, skills_dir: str, versions_dir: str = "outputs/skill_versions"):
        self.skills_dir = Path(skills_dir)
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = Path(versions_dir).parent / "skill_snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> dict[str, str]:
        skills: dict[str, str] = {}
        if not self.skills_dir.exists():
            return skills
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    skills[skill_dir.name] = skill_file.read_text(encoding="utf-8")
        return skills

    def get(self, skill_name: str) -> str | None:
        skill_file = self.skills_dir / skill_name / "SKILL.md"
        if skill_file.exists():
            return skill_file.read_text(encoding="utf-8")
        return None

    def create(self, skill_name: str, content: str, reason: str = "") -> bool:
        skill_dir = self.skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            print(f"  [SkillManager] Skill '{skill_name}' 已存在，改用 patch")
            return False
        skill_file.write_text(content, encoding="utf-8")
        self._save_version(skill_name, content, action="create", reason=reason)
        print(f"  [SkillManager] ✓ 创建 Skill: {skill_name}")
        return True

    def patch(self, skill_name: str, old_text: str, new_text: str, reason: str = "") -> bool:
        skill_file = self.skills_dir / skill_name / "SKILL.md"
        if not skill_file.exists():
            print(f"  [SkillManager] ✗ Skill '{skill_name}' 不存在，无法 patch")
            return False
        content = skill_file.read_text(encoding="utf-8")
        if old_text not in content:
            print(f"  [SkillManager] ✗ 在 '{skill_name}' 中找不到目标文本")
            return False
        new_content = self._bump_version(content.replace(old_text, new_text, 1))
        skill_file.write_text(new_content, encoding="utf-8")
        self._save_version(skill_name, new_content, action="patch", reason=reason)
        print(f"  [SkillManager] ✓ 更新 Skill: {skill_name} (reason: {reason[:50]}...)")
        return True

    def get_version_history(self, skill_name: str) -> list[dict]:
        history_file = self.versions_dir / f"{skill_name}_history.json"
        if not history_file.exists():
            return []
        return json.loads(history_file.read_text(encoding="utf-8"))

    def get_all_version_summaries(self) -> dict[str, list]:
        summaries: dict[str, list] = {}
        if not self.skills_dir.exists():
            return summaries
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                name = skill_dir.name
                history = self.get_version_history(name)
                summaries[name] = [
                    {
                        "time": h["time"],
                        "action": h["action"],
                        "reason": h.get("reason", ""),
                    }
                    for h in history
                ]
        return summaries

    def get_active_versions(self) -> dict[str, int]:
        result: dict[str, int] = {}
        if not self.skills_dir.exists():
            return result
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                name = skill_dir.name
                history = self.get_version_history(name)
                result[name] = len(history)
        return result

    def estimate_tokens(self) -> dict:
        """统计当前 Skills 体积，用于 token 消耗对比。"""
        skills = self.load_all()
        rows = []
        total_chars = 0
        total_tokens = 0
        for name, content in sorted(skills.items()):
            chars = len(content)
            tokens = max(int(chars / 1.5), int(len(content.split()) * 1.3), 1)
            rows.append({"name": name, "chars": chars, "est_tokens": tokens})
            total_chars += chars
            total_tokens += tokens
        return {
            "skills": rows,
            "count": len(rows),
            "total_chars": total_chars,
            "total_est_tokens": total_tokens,
        }

    def _save_version(self, skill_name: str, content: str, action: str, reason: str):
        history_file = self.versions_dir / f"{skill_name}_history.json"
        history: list[dict] = []
        if history_file.exists():
            history = json.loads(history_file.read_text(encoding="utf-8"))
        version_num = len(history) + 1
        history.append(
            {
                "time": datetime.now().isoformat(),
                "action": action,
                "reason": reason,
                "version": version_num,
                "content": content,
                "snapshot_file": f"skill_snapshots/{skill_name}_v{version_num}.md",
                "est_tokens": max(int(len(content) / 1.5), 1),
            }
        )
        history_file.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        snapshot_path = self.snapshots_dir / f"{skill_name}_v{version_num}.md"
        snapshot_path.write_text(
            f"<!-- {skill_name} v{version_num} | {action} | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"
            f"<!-- reason: {reason[:100]} -->\n\n{content}",
            encoding="utf-8",
        )

    def _bump_version(self, content: str) -> str:
        def increment(m):
            return f"version: {int(m.group(1)) + 1}"

        return re.sub(r"version:\s*(\d+)", increment, content)
