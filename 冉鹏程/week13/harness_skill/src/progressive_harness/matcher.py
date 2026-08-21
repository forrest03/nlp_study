"""第二阶段路由：根据精简元数据说明选择 skill。"""

from __future__ import annotations

import re

from .models import CandidateSkill, SkillMetadata

ASCII_TERM_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
CJK_RUN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
MINIMUM_TERM_LENGTH = 2


class SkillMatcher:
    """仅用可解释的词法匹配为元数据候选项排序。"""

    def select(
        self,
        request: str,
        skills: tuple[SkillMetadata, ...],
        limit: int = 3,
    ) -> tuple[CandidateSkill, ...]:
        """返回得分最高的 skill，且不加载其完整说明。

        参数：
            request: 已校验的用户意图文本。
            skills: 第一阶段发现的元数据。
            limit: 返回候选项的最大数量。

        返回：
            按得分从高到低排序的正分候选项。

        异常：
            ValueError: 请求为空或候选数量小于一时抛出。
        """
        if not request.strip():
            raise ValueError("Request must not be blank")
        if limit < 1:
            raise ValueError("Limit must be at least one")

        request_terms = self._terms(request)
        candidates = [self._candidate(skill, request_terms) for skill in skills]
        selected = [candidate for candidate in candidates if candidate.score > 0]
        selected.sort(key=lambda item: (-item.score, item.metadata.name))
        return tuple(selected[:limit])

    def _candidate(self, skill: SkillMetadata, request_terms: set[str]) -> CandidateSkill:
        """仅使用元数据文本计算一个可解释的得分。"""
        haystack = f"{skill.name} {skill.description}".lower()
        matched_terms = tuple(sorted(term for term in request_terms if term in haystack))
        phrase_score = sum(len(term) for term in matched_terms)
        name_bonus = sum(2 for term in matched_terms if term in skill.name.lower())
        return CandidateSkill(skill, float(phrase_score + name_bonus), matched_terms)

    def _terms(self, value: str) -> set[str]:
        """生成 ASCII 单词和 CJK 双字词元，用于多语言词法路由。"""
        terms = set(ASCII_TERM_PATTERN.findall(value.lower()))
        for run in CJK_RUN_PATTERN.findall(value):
            terms.update(self._cjk_bigrams(run))
        return {term for term in terms if len(term) >= MINIMUM_TERM_LENGTH}

    def _cjk_bigrams(self, value: str) -> set[str]:
        """将连续 CJK 字符拆分成可重叠的双字路由词元。"""
        if len(value) < MINIMUM_TERM_LENGTH:
            return set()
        return {value[index:index + 2] for index in range(len(value) - 1)}
