"""第一阶段发现：只读取 YAML 风格的 skill front matter。"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import SkillMetadata

LOGGER = logging.getLogger("progressive_harness.catalog")
FRONT_MATTER_DELIMITER = "---"
MAX_FRONT_MATTER_CHARACTERS = 65_536
SKILL_FILE_NAME = "SKILL.md"


class SkillCatalog:
    """发现本地 skill，并刻意避免读取说明正文。"""

    def __init__(self, skills_root: Path) -> None:
        """创建以本地 skills 目录为根的元数据目录。

        参数：
            skills_root: 包含每个 skill 子目录的目录。

        返回：
            无。
        """
        self._skills_root = skills_root.resolve()

    def discover(self) -> tuple[SkillMetadata, ...]:
        """读取每一个受支持 `SKILL.md` 文件的元数据。

        参数：
            无。

        返回：
            按名称排序的 skill；其说明正文仍未被读取。

        异常：
            FileNotFoundError: 配置的 skills 目录不存在时抛出。
            ValueError: skill 的 front matter 格式错误或字段不完整时抛出。
        """
        if not self._skills_root.is_dir():
            raise FileNotFoundError(f"Skills directory does not exist: {self._skills_root}")

        metadata = [self._read_metadata(path) for path in self._skill_files()]
        sorted_metadata = tuple(sorted(metadata, key=lambda item: item.name))
        LOGGER.info(
            "skill_catalog_discovered",
            extra={"context": {"skill_count": len(sorted_metadata)}},
        )
        return sorted_metadata

    def _skill_files(self) -> tuple[Path, ...]:
        """返回 skill 清单文件，并跳过第三方依赖目录。"""
        files = []
        for skill_file in self._skills_root.rglob(SKILL_FILE_NAME):
            if "node_modules" not in skill_file.parts:
                files.append(skill_file)
        return tuple(files)

    def _read_metadata(self, skill_file: Path) -> SkillMetadata:
        """从清单中解析大小受限的 front matter，不加载其正文。"""
        prefix = self._read_front_matter(skill_file)
        values = self._parse_front_matter(prefix, skill_file)
        name = values.get("name", "").strip()
        description = values.get("description", "").strip()
        if not name or not description:
            raise ValueError(f"Skill front matter needs name and description: {skill_file}")
        return SkillMetadata(
            name=name,
            description=description,
            version=values.get("version"),
            skill_file=skill_file.resolve(),
            root_dir=skill_file.parent.resolve(),
        )

    def _read_front_matter(self, skill_file: Path) -> str:
        """在明确的资源上限内读取开头的 YAML 块。"""
        with skill_file.open("r", encoding="utf-8") as stream:
            prefix = stream.read(MAX_FRONT_MATTER_CHARACTERS)
        lines = prefix.splitlines()
        if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
            raise ValueError(f"Skill must begin with front matter: {skill_file}")
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == FRONT_MATTER_DELIMITER:
                return "\n".join(lines[1:index])
        raise ValueError(f"Front matter exceeds limit or is not closed: {skill_file}")

    def _parse_front_matter(self, source: str, skill_file: Path) -> dict[str, str]:
        """从精简 YAML 子集中解析发现阶段所需的标量字段。"""
        values: dict[str, str] = {}
        current_key: str | None = None
        for line in source.splitlines():
            key, separator, value = line.partition(":")
            if separator and not line.startswith((" ", "\t")):
                current_key = key.strip()
                values[current_key] = value.strip().removeprefix(">-").strip()
            elif current_key and line.startswith((" ", "\t")):
                values[current_key] = f"{values[current_key]} {line.strip()}".strip()
            elif line.strip():
                raise ValueError(f"Unsupported front matter syntax: {skill_file}")
        return values
