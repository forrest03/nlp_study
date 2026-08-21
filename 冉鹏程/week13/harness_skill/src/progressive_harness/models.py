"""渐进式加载各阶段共用的不可变数据契约。"""

from dataclasses import dataclass
from pathlib import Path


class SkillNotFoundError(ValueError):
    """当请求的 skill 不在已发现目录中时抛出。"""


class InvalidReferenceError(ValueError):
    """当请求的引用文件不属于 skill 允许范围时抛出。"""


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """描述本地 skill 在启动阶段预先读取的少量 front matter 元数据。"""

    name: str
    description: str
    version: str | None
    skill_file: Path
    root_dir: Path


@dataclass(frozen=True, slots=True)
class CandidateSkill:
    """表示仅根据元数据选出的 skill，尚未读取其完整说明。"""

    metadata: SkillMetadata
    score: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """包含在 skill 被选中后才加载的完整说明。"""

    metadata: SkillMetadata
    instructions: str


@dataclass(frozen=True, slots=True)
class LoadedReference:
    """包含某个 skill 经校验且被显式请求的引用文件。"""

    skill_name: str
    name: str
    path: Path
    content: str
