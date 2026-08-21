#!/usr/bin/env python3
"""审计 PopularVideo 同步范围，并维护 Windows 成功同步基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_UPSTREAM = Path(r"D:\工作\Video Agent\code\PopularVideo")
DEFAULT_WINDOWS = Path(r"D:\工作\Video Agent\code\PopularVideoWindows")
STATE_RELATIVE = Path(".codex") / "popularvideo-sync" / "state.json"
SCHEMA_VERSION = 1

REQUIRED_REPORT_HEADINGS = (
    "版本范围与工作区",
    "上游改动全量清单",
    "Windows 同步结果",
    "手动测试清单",
    "自动验证",
    "未同步项与剩余风险",
    "提交覆盖索引",
    "覆盖核对",
    "状态记录",
)
CHANGE_REQUIRED_LABELS = (
    "分类",
    "上游依据",
    "改动前",
    "改动后",
    "Mac/共享变化",
    "服务端变化",
    "用户影响",
    "Windows 结论",
    "Windows 落点",
    "自动测试",
    "手动测试",
)
MANUAL_REQUIRED_LABELS = (
    "关联改动",
    "优先级",
    "入口",
    "前置条件",
    "测试数据",
    "是否需要真实凭据",
    "预计耗时",
    "操作步骤",
    "预期结果",
    "异常/边界",
    "失败时记录",
    "自动覆盖",
    "状态",
)
VAGUE_REPORT_PATTERNS = (
    r"测试相关功能",
    r"验证一下",
    r"确保正常",
    r"按需测试",
    r"基本没问题",
    r"大量优化",
    r"一些改动",
    r"很多改动",
)


class SyncError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SyncError(f"在 {repo} 执行 git {' '.join(args)} 失败：{detail}")
    return result


def git_root(path: Path) -> Path:
    root = run_git(path, "rev-parse", "--show-toplevel").stdout.strip()
    return Path(root).resolve()


def full_commit(repo: Path, ref: str) -> str:
    return run_git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()


def subject(repo: Path, commit: str) -> str:
    return run_git(repo, "show", "-s", "--format=%s", commit).stdout.strip()


def current_branch(repo: Path) -> str:
    result = run_git(repo, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else "(detached)"


def git_status(repo: Path) -> list[str]:
    output = run_git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout
    return [line for line in output.splitlines() if line]


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    return run_git(repo, "merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def state_path(windows: Path, override: str | None) -> Path:
    return Path(override).expanduser().resolve() if override else windows / STATE_RELATIVE


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError(f"无法读取状态文件 {path}：{exc}") from exc
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SyncError(f"状态文件 {path} 的 schema 不受支持：{data.get('schema_version')!r}")
    return data


def write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def ensure_state_repos(state: dict[str, Any], upstream: Path, windows: Path) -> None:
    expected_upstream = Path(state["upstream_repo"]).resolve()
    expected_windows = Path(state["windows_repo"]).resolve()
    if expected_upstream != upstream or expected_windows != windows:
        raise SyncError(
            "状态文件属于其他仓库："
            f"upstream={expected_upstream}, windows={expected_windows}"
        )


def remote_freshness(upstream: Path, branch: str, head: str) -> dict[str, Any]:
    if branch == "(detached)":
        return {"status": "unverified", "reason": "上游 HEAD 处于 detached 状态"}
    remote = run_git(upstream, "remote", "get-url", "origin", check=False)
    if remote.returncode:
        return {"status": "unverified", "reason": "未配置 origin"}
    query = run_git(upstream, "ls-remote", "--heads", "origin", f"refs/heads/{branch}", check=False)
    if query.returncode:
        return {"status": "unverified", "reason": query.stderr.strip() or "git ls-remote 失败"}
    line = next((item for item in query.stdout.splitlines() if item.strip()), "")
    if not line:
        return {"status": "unverified", "reason": f"origin 中不存在分支 {branch!r}"}
    remote_head = line.split()[0]
    return {
        "status": "equal" if remote_head == head else "different",
        "local_head": head,
        "remote_head": remote_head,
        "remote": remote.stdout.strip(),
        "branch": branch,
    }


def common_context(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any] | None]:
    upstream = git_root(Path(args.upstream))
    windows = git_root(Path(args.windows))
    path = state_path(windows, args.state_file)
    state = load_state(path)
    if state:
        ensure_state_repos(state, upstream, windows)
    return upstream, windows, path, state


def base_payload(upstream: Path, windows: Path, path: Path, state: dict[str, Any] | None) -> dict[str, Any]:
    upstream_head = full_commit(upstream, "HEAD")
    windows_head = full_commit(windows, "HEAD")
    upstream_status = git_status(upstream)
    windows_status = git_status(windows)
    return {
        "upstream": {
            "path": str(upstream),
            "branch": current_branch(upstream),
            "head": upstream_head,
            "subject": subject(upstream, upstream_head),
            "clean": not upstream_status,
            "status": upstream_status,
        },
        "windows": {
            "path": str(windows),
            "branch": current_branch(windows),
            "head": windows_head,
            "subject": subject(windows, windows_head),
            "clean": not windows_status,
            "status": windows_status,
        },
        "state_file": str(path),
        "state_exists": state is not None,
        "last_successful_upstream_commit": state.get("last_successful_upstream_commit") if state else None,
        "active_run": state.get("active_run") if state else None,
        "history_count": len(state.get("history", [])) if state else 0,
    }


def cmd_inspect(args: argparse.Namespace) -> dict[str, Any]:
    upstream, windows, path, state = common_context(args)
    payload = base_payload(upstream, windows, path, state)
    if args.check_remote:
        payload["remote_freshness"] = remote_freshness(
            upstream, payload["upstream"]["branch"], payload["upstream"]["head"]
        )
    return payload


def cmd_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    upstream, windows, path, state = common_context(args)
    if state:
        raise SyncError(f"状态文件已存在：{path}；bootstrap 不会覆盖它")
    commit = full_commit(upstream, args.commit)
    head = full_commit(upstream, "HEAD")
    if not is_ancestor(upstream, commit, head):
        raise SyncError(f"初始化 commit {commit} 不是上游 HEAD {head} 的祖先")
    timestamp = now_utc()
    data = {
        "schema_version": SCHEMA_VERSION,
        "upstream_repo": str(upstream),
        "windows_repo": str(windows),
        "source_branch": current_branch(upstream),
        "last_successful_upstream_commit": commit,
        "last_successful_upstream_subject": subject(upstream, commit),
        "last_successful_sync_at": timestamp,
        "last_successful_windows_commit": full_commit(windows, "HEAD"),
        "active_run": None,
        "history": [
            {
                "kind": "bootstrap",
                "recorded_at": timestamp,
                "upstream_commit": commit,
                "upstream_subject": subject(upstream, commit),
                "windows_commit": full_commit(windows, "HEAD"),
                "note": args.note,
            }
        ],
    }
    write_state(path, data)
    return {"state_file": str(path), "bootstrapped_upstream_commit": commit}


def classify_path(path: str) -> str:
    if path.startswith(("macos-app/", "shared-swift/")):
        return "mac_shared"
    if path.startswith(("src/addsubtitle/", "tests/", "deploy/", ".github/")):
        return "server"
    if path.startswith(("Dockerfile", "pyproject.toml", "uv.lock")):
        return "server"
    return "other"


def changed_files(upstream: Path, older: str, newer: str) -> list[dict[str, str]]:
    output = run_git(upstream, "diff", "--name-status", "--find-renames", f"{older}..{newer}").stdout
    result: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        status = fields[0]
        path = fields[-1]
        item = {"status": status, "path": path, "area": classify_path(path)}
        if len(fields) == 3:
            item["old_path"] = fields[1]
        result.append(item)
    return result


def commit_records(upstream: Path, older: str, newer: str) -> list[dict[str, Any]]:
    hashes = [
        line.strip()
        for line in run_git(upstream, "rev-list", "--reverse", f"{older}..{newer}").stdout.splitlines()
        if line.strip()
    ]
    records: list[dict[str, Any]] = []
    for commit in hashes:
        metadata = run_git(
            upstream,
            "show",
            "-s",
            "--date=iso-strict",
            "--format=%H%x00%ad%x00%s%x00%b",
            commit,
        ).stdout.rstrip("\n").split("\x00", 3)
        files_output = run_git(
            upstream, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", commit
        ).stdout
        paths = [line for line in files_output.splitlines() if line]
        records.append(
            {
                "commit": metadata[0],
                "date": metadata[1] if len(metadata) > 1 else "",
                "subject": metadata[2] if len(metadata) > 2 else "",
                "body": metadata[3].strip() if len(metadata) > 3 else "",
                "areas": sorted({classify_path(path) for path in paths}),
                "paths": paths,
            }
        )
    return records


def cmd_audit(args: argparse.Namespace) -> dict[str, Any]:
    upstream, windows, path, state = common_context(args)
    if not state and not args.base:
        raise SyncError("状态文件不存在；仅调查时传 --base，正式同步前请初始化已确认基线")
    older = full_commit(upstream, args.base or state["last_successful_upstream_commit"])
    newer = full_commit(upstream, args.target)
    if not is_ancestor(upstream, older, newer):
        raise SyncError(f"基线 {older} 不是目标 {newer} 的祖先")
    files = changed_files(upstream, older, newer)
    commits = commit_records(upstream, older, newer)
    payload = {
        "generated_at": now_utc(),
        "upstream_repo": str(upstream),
        "windows_repo": str(windows),
        "state_file": str(path),
        "base": {"commit": older, "subject": subject(upstream, older)},
        "target": {"commit": newer, "subject": subject(upstream, newer)},
        "commit_count": len(commits),
        "commits": commits,
        "files": files,
        "files_by_area": {
            area: [item for item in files if item["area"] == area]
            for area in ("mac_shared", "server", "other")
        },
        "diff_stat": run_git(upstream, "diff", "--stat", f"{older}..{newer}").stdout.rstrip(),
    }
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        payload["output_file"] = str(output_path)
    return payload


def require_clean_upstream(upstream: Path) -> None:
    if git_status(upstream):
        raise SyncError("上游工作树不干净；上游必须保持只读，只能同步已提交代码")


def cmd_begin(args: argparse.Namespace) -> dict[str, Any]:
    upstream, windows, path, state = common_context(args)
    if not state:
        raise SyncError("没有已初始化的成功基线，不能开始同步")
    require_clean_upstream(upstream)
    target = full_commit(upstream, args.target)
    older = full_commit(upstream, state["last_successful_upstream_commit"])
    if not is_ancestor(upstream, older, target):
        raise SyncError(f"基线 {older} 不是目标 {target} 的祖先")
    active = state.get("active_run")
    if active:
        if active.get("target_upstream_commit") == target:
            return {"state_file": str(path), "active_run": active, "resumed": True}
        raise SyncError(f"已有其他活动同步：{active.get('target_upstream_commit')}")
    windows_status = git_status(windows)
    if windows_status and not args.allow_dirty_windows:
        raise SyncError("Windows 工作树不干净；使用 --allow-dirty-windows 前必须取得用户明确授权")
    active = {
        "started_at": now_utc(),
        "base_upstream_commit": older,
        "target_upstream_commit": target,
        "target_upstream_subject": subject(upstream, target),
        "upstream_branch": current_branch(upstream),
        "windows_commit_before": full_commit(windows, "HEAD"),
        "windows_status_before": windows_status,
        "dirty_windows_authorized": bool(windows_status and args.allow_dirty_windows),
    }
    state["active_run"] = active
    write_state(path, state)
    return {"state_file": str(path), "active_run": active, "resumed": False}


def report_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def heading_blocks(text: str, prefix: str) -> list[tuple[str, str]]:
    headings = list(re.finditer(r"(?m)^#{2,3}\s+(.+?)\s*$", text))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(headings):
        title = match.group(1).strip()
        id_match = re.match(rf"({re.escape(prefix)}\d{{3}})[：:]", title)
        if not id_match:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        blocks.append((id_match.group(1), text[match.start():end]))
    return blocks


def label_value(block: str, label: str) -> str | None:
    match = re.search(rf"(?m)^-\s+{re.escape(label)}：\s*(.*?)\s*$", block)
    return match.group(1).strip() if match else None


def nested_numbered_items(block: str, label: str) -> list[str]:
    label_match = re.search(rf"(?m)^-\s+{re.escape(label)}：\s*$", block)
    if not label_match:
        return []
    next_label = re.search(r"(?m)^-\s+[^\n：]+：", block[label_match.end():])
    end = label_match.end() + next_label.start() if next_label else len(block)
    segment = block[label_match.end():end]
    return re.findall(r"(?m)^\s+\d+\.\s+\S.+$", segment)


def report_integer(text: str, label: str) -> int | None:
    match = re.search(rf"(?m)^-\s+{re.escape(label)}：`?(\d+)`?\s*$", text)
    return int(match.group(1)) if match else None


def validate_sync_report(report: Path, expected_commits: list[str]) -> dict[str, Any]:
    try:
        text = report.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SyncError(f"无法按 UTF-8 读取同步报告 {report}：{exc}") from exc

    errors: list[str] = []
    for heading in REQUIRED_REPORT_HEADINGS:
        if not re.search(rf"(?m)^##\s+{re.escape(heading)}\s*$", text):
            errors.append(f"缺少章节：## {heading}")

    placeholder = re.search(r"<[^>\n]+>|\bTODO\b|待补充|待填写|\.\.\.", text, re.IGNORECASE)
    if placeholder:
        errors.append(f"报告仍包含占位符或未完成内容：{placeholder.group(0)!r}")
    for pattern in VAGUE_REPORT_PATTERNS:
        vague = re.search(pattern, text)
        if vague:
            errors.append(f"报告包含含糊表述：{vague.group(0)!r}")

    changes = heading_blocks(text, "CHANGE-")
    manuals = heading_blocks(text, "MANUAL-")
    change_ids = [item[0] for item in changes]
    manual_ids = [item[0] for item in manuals]
    if len(change_ids) != len(set(change_ids)):
        errors.append("CHANGE 编号重复")
    expected_change_ids = [f"CHANGE-{index:03d}" for index in range(1, len(change_ids) + 1)]
    if change_ids != expected_change_ids:
        errors.append(f"CHANGE 必须从 001 连续编号且按顺序排列：实际为 {change_ids}")
    if len(manual_ids) != len(set(manual_ids)):
        errors.append("MANUAL 编号重复")
    expected_manual_ids = [f"MANUAL-{index:03d}" for index in range(1, len(manual_ids) + 1)]
    if manual_ids != expected_manual_ids:
        errors.append(f"MANUAL 必须从 001 连续编号且按顺序排列：实际为 {manual_ids}")

    if expected_commits and not changes:
        errors.append("固定范围包含 commit，但报告没有任何 CHANGE 条目")
    if not expected_commits and not changes:
        if not re.search(r"<!--\s*NO_UPSTREAM_CHANGES:\s*\S.+?-->", text):
            errors.append("没有 CHANGE 时必须写 <!-- NO_UPSTREAM_CHANGES: 具体原因 -->")

    manual_id_set = set(manual_ids)
    change_id_set = set(change_ids)
    changes_with_manual = 0
    changes_without_manual = 0
    referenced_manual_ids: set[str] = set()
    allowed_conclusions = ("已同步", "已存在", "不适用", "仅服务端", "阻塞")

    for change_id, block in changes:
        for label in CHANGE_REQUIRED_LABELS:
            value = label_value(block, label)
            if value is None:
                errors.append(f"{change_id} 缺少字段：{label}")
            elif not value:
                errors.append(f"{change_id} 字段为空：{label}")
        conclusion = label_value(block, "Windows 结论") or ""
        if conclusion and not any(item in conclusion for item in allowed_conclusions):
            errors.append(f"{change_id} 的 Windows 结论不合法：{conclusion}")
        manual_value = label_value(block, "手动测试") or ""
        references = set(re.findall(r"MANUAL-\d{3}", manual_value))
        if references:
            changes_with_manual += 1
            referenced_manual_ids.update(references)
            missing = sorted(references - manual_id_set)
            if missing:
                errors.append(f"{change_id} 引用了不存在的 MANUAL：{', '.join(missing)}")
        elif re.search(r"无（.{4,}）", manual_value):
            changes_without_manual += 1
        else:
            errors.append(f"{change_id} 必须映射 MANUAL，或写无（具体原因）")

    if not manuals and not re.search(r"<!--\s*NO_MANUAL_TESTS:\s*\S.+?-->", text):
        errors.append("没有 MANUAL 时必须写 <!-- NO_MANUAL_TESTS: 具体原因 -->")

    allowed_statuses = ("待用户手测", "Codex 已手测通过", "用户已手测通过", "不适用")
    manual_priorities: list[int] = []
    for manual_id, block in manuals:
        for label in MANUAL_REQUIRED_LABELS:
            value = label_value(block, label)
            if value is None:
                errors.append(f"{manual_id} 缺少字段：{label}")
            elif label not in ("操作步骤", "预期结果") and not value:
                errors.append(f"{manual_id} 字段为空：{label}")
        priority = label_value(block, "优先级") or ""
        priority_match = re.search(r"\bP([012])\b", priority)
        if priority and not priority_match:
            errors.append(f"{manual_id} 优先级必须是 P0/P1/P2")
        elif priority_match:
            manual_priorities.append(int(priority_match.group(1)))
        status = label_value(block, "状态") or ""
        if status and not any(item in status for item in allowed_statuses):
            errors.append(f"{manual_id} 状态不合法：{status}")
        operations = nested_numbered_items(block, "操作步骤")
        expectations = nested_numbered_items(block, "预期结果")
        if not operations:
            errors.append(f"{manual_id} 缺少编号操作步骤")
        if not expectations:
            errors.append(f"{manual_id} 缺少编号预期结果")
        if operations and expectations and len(operations) != len(expectations):
            errors.append(f"{manual_id} 操作步骤数与预期结果数不一致")
        linked_changes = set(re.findall(r"CHANGE-\d{3}", label_value(block, "关联改动") or ""))
        if not linked_changes:
            errors.append(f"{manual_id} 未关联任何 CHANGE")
        missing_changes = sorted(linked_changes - change_id_set)
        if missing_changes:
            errors.append(f"{manual_id} 关联了不存在的 CHANGE：{', '.join(missing_changes)}")

    unreferenced_manuals = sorted(manual_id_set - referenced_manual_ids)
    if unreferenced_manuals:
        errors.append(f"以下 MANUAL 未被任何 CHANGE 的手动测试字段引用：{', '.join(unreferenced_manuals)}")
    if manual_priorities != sorted(manual_priorities):
        errors.append("MANUAL 必须按 P0 → P1 → P2 排序")

    for commit in expected_commits:
        if commit not in text and commit[:8] not in text:
            errors.append(f"上游 commit 未在报告中归属：{commit}")

    declared_commit_count = report_integer(text, "审计提交数")
    declared_group_count = report_integer(text, "审计变更组数")
    declared_change_count = report_integer(text, "报告 CHANGE 条目数")
    declared_manual_mapping_count = report_integer(text, "有 MANUAL 映射的 CHANGE 条目数")
    declared_no_manual_count = report_integer(text, "无需手测且已说明原因的 CHANGE 条目数")
    unexplained_count = report_integer(text, "未解释条目数")

    expected_values = (
        ("审计提交数", declared_commit_count, len(expected_commits)),
        ("审计变更组数", declared_group_count, len(changes)),
        ("报告 CHANGE 条目数", declared_change_count, len(changes)),
        ("有 MANUAL 映射的 CHANGE 条目数", declared_manual_mapping_count, changes_with_manual),
        ("无需手测且已说明原因的 CHANGE 条目数", declared_no_manual_count, changes_without_manual),
        ("未解释条目数", unexplained_count, 0),
    )
    for label, actual, expected in expected_values:
        if actual is None:
            errors.append(f"覆盖核对缺少整数：{label}")
        elif actual != expected:
            errors.append(f"覆盖核对不一致：{label}={actual}，实际应为 {expected}")

    if errors:
        raise SyncError("同步报告校验失败：\n- " + "\n- ".join(errors))
    return {
        "change_count": len(changes),
        "manual_test_count": len(manuals),
        "changes_with_manual": changes_with_manual,
        "changes_without_manual": changes_without_manual,
        "covered_commit_count": len(expected_commits),
    }


def range_commits(upstream: Path, older: str, newer: str) -> list[str]:
    return [
        line.strip()
        for line in run_git(upstream, "rev-list", "--reverse", f"{older}..{newer}").stdout.splitlines()
        if line.strip()
    ]


def cmd_validate_report(args: argparse.Namespace) -> dict[str, Any]:
    upstream, windows, path, state = common_context(args)
    if not state and not args.base:
        raise SyncError("状态文件不存在；校验报告时请显式传 --base")
    older = full_commit(upstream, args.base or state["last_successful_upstream_commit"])
    newer = full_commit(upstream, args.target)
    if not is_ancestor(upstream, older, newer):
        raise SyncError(f"基线 {older} 不是目标 {newer} 的祖先")
    report = Path(args.report_file).expanduser().resolve()
    if not report.is_file():
        raise SyncError(f"同步报告不存在：{report}")
    inventory = validate_sync_report(report, range_commits(upstream, older, newer))
    return {
        "valid": True,
        "report_file": str(report),
        "base": older,
        "target": newer,
        "inventory": inventory,
        "state_file": str(path),
        "windows_repo": str(windows),
    }


def cmd_complete(args: argparse.Namespace) -> dict[str, Any]:
    upstream, windows, path, state = common_context(args)
    if not state or not state.get("active_run"):
        raise SyncError("没有活动同步，不能推进基线")
    require_clean_upstream(upstream)
    target = full_commit(upstream, args.target)
    active = state["active_run"]
    if active["target_upstream_commit"] != target:
        raise SyncError(f"活动目标 {active['target_upstream_commit']} 与 {target} 不一致")
    if not args.verification:
        raise SyncError("至少需要一个 --verification 结果")
    report = Path(args.report_file).expanduser().resolve()
    if not report.is_file():
        raise SyncError(f"同步报告不存在：{report}")
    expected_report_root = (windows / ".codex" / "popularvideo-sync" / "reports").resolve()
    try:
        report.relative_to(expected_report_root)
    except ValueError as exc:
        raise SyncError(f"同步报告必须位于 {expected_report_root}：{report}") from exc
    report_inventory = validate_sync_report(
        report,
        range_commits(upstream, active["base_upstream_commit"], target),
    )
    completed_at = now_utc()
    record = {
        "kind": "sync",
        "started_at": active["started_at"],
        "completed_at": completed_at,
        "from_upstream_commit": active["base_upstream_commit"],
        "to_upstream_commit": target,
        "to_upstream_subject": subject(upstream, target),
        "windows_commit_before": active["windows_commit_before"],
        "windows_commit_after": full_commit(windows, "HEAD"),
        "windows_status_before": active["windows_status_before"],
        "windows_status_after": git_status(windows),
        "windows_files_changed_by_run": sorted(set(args.windows_file or [])),
        "verification": args.verification,
        "report_file": str(report),
        "report_sha256": report_digest(report),
        "report_inventory": report_inventory,
        "summary": args.summary,
    }
    state.setdefault("history", []).append(record)
    state["last_successful_upstream_commit"] = target
    state["last_successful_upstream_subject"] = subject(upstream, target)
    state["last_successful_sync_at"] = completed_at
    state["last_successful_windows_commit"] = record["windows_commit_after"]
    state["active_run"] = None
    write_state(path, state)
    return {
        "state_file": str(path),
        "last_successful_upstream_commit": target,
        "history_count": len(state["history"]),
        "report_file": str(report),
    }


def cmd_abort(args: argparse.Namespace) -> dict[str, Any]:
    upstream, windows, path, state = common_context(args)
    if not state or not state.get("active_run"):
        raise SyncError("没有活动同步")
    active = state["active_run"]
    state.setdefault("history", []).append(
        {
            "kind": "aborted",
            "started_at": active["started_at"],
            "aborted_at": now_utc(),
            "from_upstream_commit": active["base_upstream_commit"],
            "to_upstream_commit": active["target_upstream_commit"],
            "reason": args.reason,
        }
    )
    state["active_run"] = None
    write_state(path, state)
    return {"state_file": str(path), "aborted_target": active["target_upstream_commit"]}


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM), help="PopularVideo 上游 Git 仓库")
    parser.add_argument("--windows", default=str(DEFAULT_WINDOWS), help="PopularVideoWindows Git 仓库")
    parser.add_argument("--state-file", help="覆盖默认的忽略状态文件路径")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="显示仓库与已保存同步状态")
    add_common(inspect_parser)
    inspect_parser.add_argument("--check-remote", action="store_true", help="用 ls-remote 比较本地 HEAD 与 origin")
    inspect_parser.set_defaults(handler=cmd_inspect)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="记录已确认的既有成功基线")
    add_common(bootstrap_parser)
    bootstrap_parser.add_argument("--commit", required=True, help="已确认的上游 commit")
    bootstrap_parser.add_argument("--note", required=True, help="基线证据或用户确认")
    bootstrap_parser.set_defaults(handler=cmd_bootstrap)

    audit_parser = subparsers.add_parser("audit", help="收集固定范围内的 commit 与路径证据")
    add_common(audit_parser)
    audit_parser.add_argument("--base", help="状态缺失时仅用于调查的基线")
    audit_parser.add_argument("--target", required=True, help="固定的上游目标 commit")
    audit_parser.add_argument("--output", help="可选 JSON 输出路径")
    audit_parser.set_defaults(handler=cmd_audit)

    begin_parser = subparsers.add_parser("begin", help="记录一次活动同步")
    add_common(begin_parser)
    begin_parser.add_argument("--target", required=True, help="固定的上游目标 commit")
    begin_parser.add_argument("--allow-dirty-windows", action="store_true", help="仅在用户明确授权后使用")
    begin_parser.set_defaults(handler=cmd_begin)

    validate_parser = subparsers.add_parser("validate-report", help="校验中文改动清单与手动测试清单")
    add_common(validate_parser)
    validate_parser.add_argument("--report-file", required=True, help="待校验的 Markdown 报告")
    validate_parser.add_argument("--base", help="报告对应的上游基线；默认读取状态")
    validate_parser.add_argument("--target", required=True, help="报告对应的固定目标 commit")
    validate_parser.set_defaults(handler=cmd_validate_report)

    complete_parser = subparsers.add_parser("complete", help="实现、验证和报告通过后推进基线")
    add_common(complete_parser)
    complete_parser.add_argument("--target", required=True, help="固定的上游目标 commit")
    complete_parser.add_argument("--report-file", required=True, help="已完成的 Markdown 报告")
    complete_parser.add_argument("--summary", required=True, help="具体的同步结果摘要")
    complete_parser.add_argument("--verification", action="append", help="每个验证结果重复传入一次")
    complete_parser.add_argument("--windows-file", action="append", help="本轮改动的每个 Windows 文件重复传入一次")
    complete_parser.set_defaults(handler=cmd_complete)

    abort_parser = subparsers.add_parser("abort", help="放弃活动同步且不推进基线")
    add_common(abort_parser)
    abort_parser.add_argument("--reason", required=True)
    abort_parser.set_defaults(handler=cmd_abort)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = args.handler(args)
    except SyncError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
