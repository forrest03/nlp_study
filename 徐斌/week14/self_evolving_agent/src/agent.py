"""
知识点匹配 Agent：仅依据当前 Skills 判断习题对应知识点。

契约：
  - 能匹配 → 只输出知识点名称（与 Skill 中登记的名称一致）
  - 不能匹配 → 仅回答「无法匹配知识点」
"""

from __future__ import annotations

from llm_config import get_chat_client
from skill_manager import SkillManager

SYSTEM_TEMPLATE = """你是初中数学习题「知识点匹配」助手。

你的全部依据来自下方技能文档。禁止凭预训练常识猜测未登记的知识点名称。

## 回答规则（严格遵守）
- 【能匹配】技能文档覆盖该题型：只输出知识点名称本身（不要解题、不要解释）。
- 【不能匹配】技能文档未覆盖：**仅回答一句**「无法匹配知识点」。

{skills_section}
"""

SKILLS_SECTION_TEMPLATE = """## 当前技能库（共 {count} 个）

{skills_content}
"""


class TopicMatchAgent:
    def __init__(self, skill_manager: SkillManager, nudge_interval: int = 0):
        self.skill_manager = skill_manager
        self.nudge_interval = nudge_interval
        self._iters_since_nudge = 0
        self.conversation_history: list[dict] = []
        self.client, self.model = get_chat_client()
        self.last_usage: dict = {}

    def answer(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=120,
        )
        answer_text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        self.conversation_history.append(
            {
                "question": question,
                "answer": answer_text,
                "skills_used": list(self.skill_manager.load_all().keys()),
                "usage": self.last_usage,
            }
        )
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

        self._iters_since_nudge += 1
        return answer_text

    def should_trigger_nudge(self) -> bool:
        if self.nudge_interval > 0 and self._iters_since_nudge >= self.nudge_interval:
            self._iters_since_nudge = 0
            return True
        return False

    def reset_nudge_counter(self) -> None:
        self._iters_since_nudge = 0

    def _build_system_prompt(self) -> str:
        skills = self.skill_manager.load_all()
        if not skills:
            skills_section = "（暂无技能文档）"
        else:
            parts = [f"### 技能：{name}\n{content}" for name, content in sorted(skills.items())]
            skills_section = SKILLS_SECTION_TEMPLATE.format(
                count=len(skills),
                skills_content="\n\n---\n\n".join(parts),
            )
        return SYSTEM_TEMPLATE.format(skills_section=skills_section)
