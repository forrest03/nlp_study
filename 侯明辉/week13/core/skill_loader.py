"""
skill_loader.py — 渐进式加载决策

核心算法：
  1. 关键词初筛：拿 user 输入对所有 Skill 的 trigger 关键词做正则/子串匹配
  2. LLM 二次确认：拿候选列表 + 用户输入，让 LLM 输出应该加载哪些 Skill
  3. 加载选中的 Skill 完整内容

为什么不直接用 LLM 每次挑？
  - 零成本过滤 95% 的不相关 Skill（关键词命中只有一两个）
  - MEMORY.md 索引 < 200 tokens，关键词匹配 O(N) 极快
  - LLM 二次确认是兜底，处理「没明确触发词但语义相关」的情况
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from core.skill_registry import SkillMeta, SkillRegistry, SkillFull


# ── 关键词匹配 ─────────────────────────────────────────────────────────────

def keyword_match(user_input: str, metas: Sequence[SkillMeta]) -> list[str]:
    """对每个 Skill 的 trigger 关键词做子串匹配（大小写不敏感）"""
    text = user_input.lower()
    hits: list[str] = []
    for m in metas:
        if not m.trigger:
            continue
        keywords = [k.strip().lower() for k in re.split(r"[|,]", m.trigger) if k.strip()]
        if any(kw in text for kw in keywords):
            hits.append(m.name)
    return hits


# ── 加载器 ────────────────────────────────────────────────────────────────

@dataclass
class SkillLoader:
    registry: SkillRegistry

    def _llm_confirm(
        self,
        user_input: str,
        candidates: list[SkillMeta],
        client,
        model: str,
    ) -> list[str]:
        """让 LLM 从候选 Skill 中挑真正需要加载的"""
        cand_lines = [f"- {m.name}: {m.description}" for m in candidates]
        prompt = (
            f"你是 Skill 路由器。用户说：「{user_input}」\n\n"
            f"候选 Skill：\n{chr(10).join(cand_lines)}\n\n"
            f"请判断哪些 Skill 真正需要被加载来处理这个请求。\n"
            f"- 只输出 JSON 数组，例如：[\"weather\"]\n"
            f"- 如果都不需要，输出 []\n"
            f"- 不要任何解释文字\n"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            # 提取 JSON 数组
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if not match:
                return []
            arr = json.loads(match.group(0))
            if not isinstance(arr, list):
                return []
            valid_names = {m.name for m in candidates}
            return [n for n in arr if isinstance(n, str) and n in valid_names]
        except Exception:
            # LLM 调用失败 → 降级到关键词命中
            return [m.name for m in candidates]

    def select(
        self,
        user_input: str,
        client,
        model: str,
        provider: str = "",
    ) -> list[str]:
        """主入口：决定要加载哪些 Skill 名字"""
        candidates = self.registry.list_all()
        if not candidates:
            return []
        # 1. 关键词初筛
        keyword_hits = keyword_match(user_input, candidates)
        if not keyword_hits:
            return []
        # 2. 只对关键词命中的 Skill 做 LLM 二次确认
        hit_metas = [m for m in candidates if m.name in keyword_hits]
        if len(hit_metas) == 1:
            # 单一命中，跳过 LLM 直通
            return [hit_metas[0].name]
        return self._llm_confirm(user_input, hit_metas, client, model)

    def load_for_input(
        self,
        user_input: str,
        client,
        model: str,
        provider: str = "",
    ) -> list[SkillFull]:
        """select + 加载完整内容"""
        names = self.select(user_input, client, model, provider)
        return self.registry.load_multiple(names)
