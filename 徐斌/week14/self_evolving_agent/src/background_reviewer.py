"""
后台回顾 Agent：仅根据失败样本，最小改动 create/patch 知识点匹配 Skills。
同时约束 Skill 体积，避免堆砌代表题导致 token 膨胀。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_config import get_chat_client
from skill_manager import SkillManager

REVIEWER_SYSTEM = """你是「习题→知识点」技能优化专家。

给你的全部是 Agent 最近一轮中**答错或推脱**的样本。请用最小改动修复它们。

## 核心原则（严格遵守）

1. **仅修复观察到的失败**：不要扩展到样本里没出现的知识点
2. **最小改动优先**：能 patch 就不要 create；old_text 只含要改的那几行
3. **按失败频次只修 1~2 类**：留出进化梯度
4. **控制 token**：Skill 写成决策 SOP，不要整段粘贴长题干；
   每个知识点最多 2 条短特征/触发词 + 1 条例子（可截断到 80 字内）
5. **知识点名称必须与判定标准完全一致**（一字不差）

你拥有完整判定标准文档，仅用于核对正确知识点名称与题型特征。

## 判定标准（权威）
{policies}

## 当前已有 Skill 摘要
{current_skills_summary}

## 输出格式
{{
  "analysis": "本轮失败 N 条，主要失败类型是 XXX",
  "actions": [
    {{"action": "create", "skill_name": "...", "reason": "修复哪类失败",
      "content": "完整SKILL.md（含frontmatter，控制在约80行内）"}},
    {{"action": "patch",  "skill_name": "...", "reason": "修复哪类失败",
      "old_text": "精确的原始文本", "new_text": "替换文本"}}
  ]
}}

只输出 JSON。若失败模式不清，可返回 0 条 action。"""

REVIEWER_USER = """## 本轮失败样本（共 {n} 条）

{history_text}

## 当前 Skill 完整内容
{current_skills_full}

按核心原则给出最小必要的修复方案。"""


class BackgroundReviewer:
    def __init__(self, policies_path: str, skill_manager: SkillManager):
        self.policies = Path(policies_path).read_text(encoding="utf-8")
        self.skill_manager = skill_manager
        self.last_analysis = ""
        self.client, self.model = get_chat_client()

    def review(self, failed_turns: list[dict]) -> list[dict]:
        if not failed_turns:
            return []

        current_skills = self.skill_manager.load_all()
        skills_summary = (
            "\n".join(
                f"- {name}: {self._extract_description(content)}"
                for name, content in sorted(current_skills.items())
            )
            or "（暂无已有Skill）"
        )
        skills_full = (
            "\n\n---\n\n".join(
                f"### {name}\n{content}"
                for name, content in sorted(current_skills.items())
            )
            or "（暂无已有Skill）"
        )

        system_msg = REVIEWER_SYSTEM.format(
            policies=self.policies,
            current_skills_summary=skills_summary,
        )
        user_msg = REVIEWER_USER.format(
            n=len(failed_turns),
            history_text=self._format_history(failed_turns),
            current_skills_full=skills_full,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=3500,
        )
        return self._parse_actions((response.choices[0].message.content or "").strip())

    def _format_history(self, turns: list[dict]) -> str:
        lines = []
        for i, t in enumerate(turns, 1):
            q = t["question"]
            if len(q) > 280:
                q = q[:280] + "…"
            a = t["answer"]
            if len(a) > 120:
                a = a[:120] + "…"
            lines.append(f"[{i}] 用户: {q}")
            lines.append(f"    Agent: {a}")
            if t.get("fail_reason"):
                lines.append(f"    ✗ 判定：{t['fail_reason']}")
            if t.get("topic_name"):
                lines.append(f"    ✓ 期望知识点：{t['topic_name']}")
        return "\n".join(lines)

    def _extract_description(self, content: str) -> str:
        m = re.search(r"description:\s*(.+)", content)
        return m.group(1).strip() if m else "(无描述)"

    def _parse_actions(self, raw: str) -> list[dict]:
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                print(f"  [Reviewer] 无法提取 JSON，原始输出：{raw[:200]}")
                self.last_analysis = ""
                return []
            data = json.loads(json_match.group())
            self.last_analysis = data.get("analysis", "")
            print(f"  [Reviewer] 分析：{self.last_analysis[:120]}")
            return data.get("actions", [])
        except json.JSONDecodeError as e:
            print(f"  [Reviewer] JSON 解析失败: {e}\n原始: {raw[:300]}")
            self.last_analysis = ""
            return []
