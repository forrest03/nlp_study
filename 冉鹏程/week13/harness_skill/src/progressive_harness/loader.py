"""第三、四阶段：加载选中的 skill 及其安全引用文件。"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import InvalidReferenceError, LoadedReference, LoadedSkill, SkillMetadata, SkillNotFoundError

LOGGER = logging.getLogger("progressive_harness.loader")
REFERENCES_DIRECTORY_NAME = "references"
REFERENCE_SUFFIX = ".md"


class SkillLoader:
    """加载选中的说明文件和被显式批准的引用文件。"""

    def __init__(self, skills: tuple[SkillMetadata, ...]) -> None:
        """创建受已发现元数据快照约束的加载器。

        参数：
            skills: 由 `SkillCatalog.discover` 获得的 skill。

        返回：
            无。
        """
        self._by_name = {skill.name: skill for skill in skills}

    def load_skill(self, skill_name: str) -> LoadedSkill:
        """仅在 skill 名称被选中后加载其完整说明文件。

        参数：
            skill_name: 已发现元数据快照中的 skill 名称。

        返回：
            所选 skill 的元数据和完整 Markdown 说明。

        异常：
            SkillNotFoundError: 未发现 `skill_name` 时抛出。
            OSError: 无法读取说明文件时抛出。
        """
        metadata = self._metadata_for(skill_name)
        instructions = self._read_text(metadata.skill_file)
        LOGGER.info(
            "skill_instructions_loaded",
            extra={"context": {"skill_name": metadata.name, "character_count": len(instructions)}},
        )
        return LoadedSkill(metadata=metadata, instructions=instructions)

    def load_reference(self, skill_name: str, reference_name: str) -> LoadedReference:
        """校验本地路径边界后，加载一个被批准的引用文件。

        参数：
            skill_name: 已发现元数据快照中的 skill 名称。
            reference_name: `references/` 下 Markdown 文件的基本名称。

        返回：
            请求的引用内容及其解析后的路径。

        异常：
            SkillNotFoundError: 未发现 `skill_name` 时抛出。
            InvalidReferenceError: 引用不安全或不可用时抛出。
            OSError: 无法读取被批准的引用时抛出。
        """
        metadata = self._metadata_for(skill_name)
        reference_path = self._reference_path(metadata, reference_name)
        content = self._read_text(reference_path)
        LOGGER.info(
            "skill_reference_loaded",
            extra={"context": {"skill_name": skill_name, "reference_name": reference_name}},
        )
        return LoadedReference(skill_name, reference_name, reference_path, content)

    def _metadata_for(self, skill_name: str) -> SkillMetadata:
        """根据发现快照解析选中的 skill 名称。"""
        metadata = self._by_name.get(skill_name)
        if metadata is None:
            raise SkillNotFoundError(f"Unknown skill: {skill_name}")
        return metadata

    def _reference_path(self, metadata: SkillMetadata, reference_name: str) -> Path:
        """仅当引用文件仍位于 skill 的 references 目录下时才解析它。"""
        candidate_name = Path(reference_name)
        if candidate_name.name != reference_name or candidate_name.suffix != REFERENCE_SUFFIX:
            raise InvalidReferenceError("Reference must be a Markdown basename")
        references_dir = (metadata.root_dir / REFERENCES_DIRECTORY_NAME).resolve()
        candidate_path = (references_dir / candidate_name).resolve()
        if references_dir not in candidate_path.parents or not candidate_path.is_file():
            raise InvalidReferenceError(f"Reference is not available: {reference_name}")
        return candidate_path

    def _read_text(self, path: Path) -> str:
        """读取 UTF-8 文本，并将调用方可见的 IO 异常显式向上传递。"""
        with path.open("r", encoding="utf-8") as stream:
            return stream.read()
