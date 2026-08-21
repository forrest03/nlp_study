"""
skill 管理模块。

目标：
1. 自动扫描 skills 目录下的技能目录
2. 读取每个技能目录中的 SKILL.md
3. 根据当前问题挑选最相关的 skill
4. 只在命中时加载 skill 完整内容，避免一次把所有 skill 都塞进 prompt
5. 记录本轮已经加载过哪些 skill，方便做渐进式加载
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).parent
SKILLS_DIR = BASE_DIR / "skills"
GENERIC_KEYWORDS = {
    "使用", "用户", "内容", "当前", "生成", "文件", "目录", "数据",
    "路径", "脚本", "工具", "输出", "输入", "类型", "规则", "方式",
    "流程", "支持", "保存", "创建", "打开", "处理", "一个", "可以",
    "skill", "use", "user", "data", "file", "files", "path", "paths",
    "script", "scripts", "tool", "tools", "output", "input", "json",
    "en", "zh",
}


class SkillManager:
    """skill 管理器。"""

    def __init__(self):
        self.cache: dict[str, str] = {}
        self.loaded_skills: list[str] = []
        self.skill_manifest = self._discover_skills()
        self.manifest_map = {item["name"]: item for item in self.skill_manifest}

    def _discover_skills(self) -> list[dict[str, Any]]:
        """扫描 skills 目录，自动发现 skill。"""
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []

        for skill_dir in sorted(SKILLS_DIR.iterdir()):
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            try:
                skill_meta = self._parse_skill_file(skill_file)
                manifest.append(skill_meta)
            except Exception:
                # 单个 skill 解析失败时，别拖垮整个 agent
                continue

        return manifest

    def _parse_front_matter(self, text: str) -> tuple[dict[str, str], str]:
        """解析 markdown 顶部的 yaml 风格 front matter。"""
        stripped = text.lstrip()
        if not stripped.startswith("---"):
            return {}, text

        lines = stripped.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, text

        end_idx = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break

        if end_idx is None:
            return {}, text

        meta_lines = lines[1:end_idx]
        body = "\n".join(lines[end_idx + 1 :])
        meta: dict[str, str] = {}
        current_key = None
        current_value_lines: list[str] = []

        for raw_line in meta_lines:
            line = raw_line.rstrip()
            if not line.strip():
                continue

            if re.match(r"^[A-Za-z0-9_-]+\s*:", line):
                if current_key is not None:
                    meta[current_key] = "\n".join(current_value_lines).strip()
                    current_value_lines = []

                key, value = line.split(":", 1)
                current_key = key.strip()
                value = value.strip()
                if value in {">", "|", ">-", "|-"}:
                    current_value_lines = []
                else:
                    current_value_lines = [value]
            elif current_key is not None:
                current_value_lines.append(line.strip())

        if current_key is not None:
            meta[current_key] = "\n".join(current_value_lines).strip()

        return meta, body

    def _extract_keywords(self, skill_name: str, description: str, body: str) -> list[str]:
        """从 skill 文本里做一个宽松的关键词抽取。"""
        keywords: list[str] = []

        def add_kw(value: str):
            value = value.strip().strip("\"'`").lower()
            if len(value) < 2:
                return
            if value in GENERIC_KEYWORDS:
                return
            if value not in keywords:
                keywords.append(value)

        add_kw(skill_name)
        for chunk in [description, body]:
            for quoted in re.findall(r"[\"“”']([^\"“”'\n]{2,40})[\"“”']", chunk):
                add_kw(quoted)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", chunk):
                add_kw(token)
            for phrase in re.findall(r"[\u4e00-\u9fff]{2,12}", chunk):
                add_kw(phrase)

        return keywords[:80]

    def _extract_tools(self, body: str) -> list[str]:
        """从 skill 文本里提取可能提到的工具名。"""
        tools: list[str] = []
        tool_candidates = [
            "query_geo",
            "query_weather",
            "company_lookup",
            "stock_info",
            "get_system_info",
            "run_local_script",
            "list_directory",
            "read_text_file",
            "write_text_file",
            "open_file",
            "update_soul_memory",
            "update_user_memory",
            "update_long_term_memory",
        ]
        for tool_name in tool_candidates:
            if tool_name in body and tool_name not in tools:
                tools.append(tool_name)
        return tools

    def _parse_skill_file(self, skill_file: Path) -> dict[str, Any]:
        text = skill_file.read_text(encoding="utf-8")
        front_matter, body = self._parse_front_matter(text)

        skill_name = (
            front_matter.get("name")
            or skill_file.parent.name
        ).strip()
        summary = (
            front_matter.get("description")
            or self._extract_summary_from_body(body)
            or f"{skill_name} skill"
        ).strip()

        return {
            "name": skill_name,
            "summary": re.sub(r"\s+", " ", summary),
            "keywords": self._extract_keywords(skill_name, summary, body),
            "tools": self._extract_tools(body),
            "file_path": str(skill_file),
            "dir_path": str(skill_file.parent),
        }

    def _extract_summary_from_body(self, body: str) -> str:
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            return stripped[:160]
        return ""

    def get_manifest_text(self) -> str:
        """给 prompt 用的 skill 摘要。"""
        if not self.skill_manifest:
            return "当前没有发现可用 skill。"

        lines = []
        for item in self.skill_manifest:
            tools_text = "、".join(item["tools"]) if item["tools"] else "无专属工具"
            lines.append(f"- {item['name']}: {item['summary']} 可用工具: {tools_text}")
        return "\n".join(lines)

    def list_skills(self) -> list[str]:
        """返回所有 skill 名称。"""
        return [item["name"] for item in self.skill_manifest]

    def get_skill_meta(self, skill_name: str) -> dict[str, Any] | None:
        """查询单个 skill 元信息。"""
        return self.manifest_map.get(skill_name)

    def _score_skill(self, text: str, item: dict[str, Any]) -> int:
        if not text:
            return 0

        text_lower = text.lower()
        score = 0

        for kw in item.get("keywords", []):
            if kw and kw in text_lower:
                score += max(1, min(len(kw), 6))

        summary = item.get("summary", "").lower()
        if summary and any(part in summary for part in text_lower.split() if len(part) > 1):
            score += 2

        name = item.get("name", "").lower()
        if name and name in text_lower:
            score += 4

        return score

    def select_skills(self, text: str, top_k: int = 2) -> list[str]:
        """
        根据当前输入粗筛 skill。

        这里先用目录扫描 + 宽松关键词命中。
        """
        scores = []
        for item in self.skill_manifest:
            score = self._score_skill(text, item)
            if score > 0:
                scores.append((score, item["name"]))

        scores.sort(key=lambda x: x[0], reverse=True)
        if not scores:
            return []

        top_score = scores[0][0]
        min_score = max(3, top_score // 2)
        filtered = [(score, name) for score, name in scores if score >= min_score]
        return [name for _, name in filtered[:top_k]]

    def load_skill(self, skill_name: str) -> str:
        """加载单个 skill 完整内容，带缓存。"""
        if skill_name in self.cache:
            return self.cache[skill_name]

        meta = self.manifest_map.get(skill_name)
        if meta is None:
            raise ValueError(f"skill 不存在：{skill_name}")

        file_path = Path(meta["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"skill 文件不存在：{file_path}")

        content = file_path.read_text(encoding="utf-8")
        prefix = (
            f"[skill_name] {meta['name']}\n"
            f"[skill_dir] {meta['dir_path']}\n"
            "注意：这个 skill 目录下可能还有 scripts/、references/、data/ 等资源文件，"
            "需要时应基于 skill_dir 自己去定位。\n\n"
        )
        final_content = prefix + content
        self.cache[skill_name] = final_content
        return final_content

    def load_skills(self, skill_names: list[str]) -> str:
        """批量加载多个 skill 的完整内容。"""
        if not skill_names:
            return ""

        blocks = []
        for skill_name in skill_names:
            blocks.append(f"[{skill_name}]\n{self.load_skill(skill_name)}")
        return "\n\n".join(blocks)

    def get_new_skills_to_load(self, text: str, top_k: int = 2) -> list[str]:
        """只返回这次命中、但此前没加载过的 skill。"""
        candidates = self.select_skills(text, top_k=top_k)
        return [name for name in candidates if name not in self.loaded_skills]

    def mark_skills_loaded(self, skill_names: list[str]):
        """把已经加载过的 skill 记下来。"""
        for name in skill_names:
            if name not in self.loaded_skills:
                self.loaded_skills.append(name)

    def progressive_load(self, text: str, top_k: int = 2) -> dict[str, Any]:
        """
        对外暴露的渐进式加载接口。

        返回：
        - matched_skills: 当前命中的 skill
        - new_skills: 这次新加载的 skill
        - loaded_prompt: 新加载 skill 的完整内容
        """
        matched_skills = self.select_skills(text, top_k=top_k)
        new_skills = [name for name in matched_skills if name not in self.loaded_skills]
        loaded_prompt = self.load_skills(new_skills) if new_skills else ""
        active_prompt = self.load_skills(matched_skills) if matched_skills else ""
        self.mark_skills_loaded(new_skills)

        return {
            "matched_skills": matched_skills,
            "new_skills": new_skills,
            "loaded_prompt": loaded_prompt,
            "active_prompt": active_prompt,
            "loaded_skills": self.loaded_skills.copy(),
        }

    def reset_loaded_skills(self):
        """开启新会话时可以重置已加载状态。"""
        self.loaded_skills = []


if __name__ == "__main__":
    manager = SkillManager()
    print("当前 skill 清单：")
    print(manager.get_manifest_text())
    print()
    print("测试渐进式加载：")
    print(manager.progressive_load("帮我画一个系统架构图"))
    print(manager.progressive_load("给我做一张 crazy 的 flash card"))
