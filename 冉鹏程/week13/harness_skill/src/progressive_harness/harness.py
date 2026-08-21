"""协调渐进式 skill 加载各阶段的应用服务。"""

from __future__ import annotations

import logging
from pathlib import Path

from .catalog import SkillCatalog
from .loader import SkillLoader
from .matcher import SkillMatcher
from .models import CandidateSkill, LoadedReference, LoadedSkill, SkillMetadata

LOGGER = logging.getLogger("progressive_harness.harness")
MAX_REQUEST_CHARACTERS = 4_000


class ProgressiveHarness:
    """协调发现、元数据路由、说明加载和引用文件读取。"""

    def __init__(self, skills_root: Path) -> None:
        """为给定的本地 skill 目录创建 Harness。

        参数：
            skills_root: 包含所有本地 skill 的目录。

        返回：
            无。
        """
        self._catalog = SkillCatalog(skills_root)
        self._matcher = SkillMatcher()

    def discover(self) -> tuple[SkillMetadata, ...]:
        """通过仅读取本地 skill 元数据来执行第一阶段。

        参数：
            无。

        返回：
            仅包含元数据的本地 skill 目录快照。

        异常：
            FileNotFoundError: skills 根目录不存在时抛出。
            ValueError: skill 清单格式错误时抛出。
        """
        return self._catalog.discover()

    def select(
        self,
        request: str,
        skills: tuple[SkillMetadata, ...],
        limit: int = 3,
    ) -> tuple[CandidateSkill, ...]:
        """通过元数据选择候选 skill 来执行第二阶段。

        参数：
            request: 不会写入日志的用户意图。
            skills: `discover` 返回的元数据快照。
            limit: 返回匹配 skill 的最大数量。

        返回：
            按路由得分从高到低排序的匹配 skill。

        异常：
            ValueError: 输入为空、过长或候选数量无效时抛出。
        """
        self._validate_request(request)
        candidates = self._matcher.select(request, skills, limit)
        LOGGER.info(
            "skill_candidates_selected",
            extra={"context": {"candidate_count": len(candidates), "request_length": len(request)}},
        )
        return candidates

    def load_skill(self, skill_name: str, skills: tuple[SkillMetadata, ...]) -> LoadedSkill:
        """加载一个已选 skill 的说明，以执行第三阶段。

        参数：
            skill_name: 第二阶段选中的 skill 名称。
            skills: `discover` 返回的元数据快照。

        返回：
            所选 skill 完整加载后的 Markdown 说明。

        异常：
            SkillNotFoundError: 所选名称不在快照中时抛出。
            OSError: 无法读取说明文件时抛出。
        """
        return SkillLoader(skills).load_skill(skill_name)

    def load_reference(
        self,
        skill_name: str,
        reference_name: str,
        skills: tuple[SkillMetadata, ...],
    ) -> LoadedReference:
        """加载一个被显式请求的引用文件，以执行第四阶段。

        参数：
            skill_name: 已加载说明的 skill 名称。
            reference_name: 调用方请求的 Markdown 基本文件名。
            skills: `discover` 返回的元数据快照。

        返回：
            所选 skill 中被批准引用文件的内容。

        异常：
            SkillNotFoundError: 未发现该 skill 时抛出。
            InvalidReferenceError: 引用不安全或未在本地列出时抛出。
            OSError: 无法读取引用文件时抛出。
        """
        return SkillLoader(skills).load_reference(skill_name, reference_name)

    def _validate_request(self, request: str) -> None:
        """在路由前拒绝为空或异常过长的外部请求输入。"""
        if not request.strip():
            raise ValueError("Request must not be blank")
        if len(request) > MAX_REQUEST_CHARACTERS:
            raise ValueError(f"Request exceeds {MAX_REQUEST_CHARACTERS} characters")
