"""用于观察每个渐进式加载阶段的命令行接口。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .harness import ProgressiveHarness
from .logging_utils import configure_logging
from .models import CandidateSkill, LoadedReference, LoadedSkill, SkillMetadata

LOGGER = logging.getLogger("progressive_harness.cli")
DEFAULT_MAX_CANDIDATES = 3


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。

    参数：
        无。

    返回：
        用于列出目录和处理一次渐进式加载请求的解析器。
    """
    parser = argparse.ArgumentParser(description="Progressively load local agent skills")
    parser.add_argument("request", nargs="?", help="Intent used to route a local skill")
    parser.add_argument("--skills-dir", type=Path, default=_default_skills_dir())
    parser.add_argument("--list", action="store_true", help="Print metadata only and exit")
    parser.add_argument("--reference", action="append", default=[], metavar="FILE.md")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    parser.add_argument("--verbose", action="store_true", help="Emit structured lifecycle logs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行 Harness CLI。

    参数：
        argv: 可选的参数序列，不包含程序名称。

    返回：
        进程退出状态；成功时为零，校验或 IO 错误时为非零。
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        return _run(args)
    except (OSError, ValueError) as error:
        LOGGER.error("harness_request_failed", extra={"context": {"error_type": type(error).__name__}})
        print(f"Error: {error}")
        return 2


def _run(args: argparse.Namespace) -> int:
    """执行请求的 CLI 路径，且不记录外部请求内容。"""
    harness = ProgressiveHarness(args.skills_dir)
    metadata = harness.discover()
    if args.list:
        _print_metadata(metadata, args.json)
        return 0
    if args.request is None:
        raise ValueError("Provide a request or use --list")

    candidates = harness.select(args.request, metadata, args.max_candidates)
    if not candidates:
        _print_no_match(args.json)
        return 1

    selected = candidates[0]
    loaded_skill = harness.load_skill(selected.metadata.name, metadata)
    references = tuple(
        harness.load_reference(selected.metadata.name, reference_name, metadata)
        for reference_name in args.reference
    )
    _print_result(metadata, candidates, loaded_skill, references, args.json)
    return 0


def _default_skills_dir() -> Path:
    """返回示例项目使用的仓库内 skills 目录。"""
    return Path(__file__).resolve().parents[2] / "skills"


def _print_metadata(metadata: tuple[SkillMetadata, ...], as_json: bool) -> None:
    """打印仅包含元数据的发现结果，不读取 skill 说明。"""
    payload = [{"name": item.name, "description": item.description, "version": item.version} for item in metadata]
    if as_json:
        print(json.dumps({"stage": "metadata", "skills": payload}, ensure_ascii=False, indent=2))
        return
    for item in payload:
        print(f"{item['name']}\t{item['description']}")


def _print_no_match(as_json: bool) -> None:
    """为人工和机器调用方输出一致的无匹配结果。"""
    if as_json:
        print(json.dumps({"stage": "selection", "candidates": []}, ensure_ascii=False))
        return
    print("No matching skill found from metadata.")


def _print_result(
    metadata: tuple[SkillMetadata, ...],
    candidates: tuple[CandidateSkill, ...],
    loaded_skill: LoadedSkill,
    references: tuple[LoadedReference, ...],
    as_json: bool,
) -> None:
    """输出全部加载阶段，仅包含被显式请求的引用文件。"""
    payload = {
        "metadata": [{"name": item.name, "description": item.description} for item in metadata],
        "candidates": [
            {"name": item.metadata.name, "score": item.score, "matched_terms": item.matched_terms}
            for item in candidates
        ],
        "loaded_skill": {"name": loaded_skill.metadata.name, "instructions": loaded_skill.instructions},
        "loaded_references": [
            {"name": item.name, "content": item.content} for item in references
        ],
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"Selected skill: {loaded_skill.metadata.name}")
    print(f"Candidates: {', '.join(item.metadata.name for item in candidates)}")
    print(f"Instruction characters loaded: {len(loaded_skill.instructions)}")
    print(f"References loaded: {', '.join(item.name for item in references) or 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
