"""bash_tool.py - 受限 Bash 执行器。

layer-2 兜底校验（FR-006）+ subprocess(shell=False) 执行（无注入面）。
- 组合命令：`;` `&&` `||` `&` 顺序执行；`|` 用 Popen 串联（research D2）。
- cd：stateful 跟踪 cwd，不 spawn 子进程（F2：使 `cd dir; ls` 生效）。
"""
from __future__ import annotations

import os
import subprocess

from .logging import log_error
from .whitelist import (BLOCK_MESSAGE, BlockResult, Reason, Whitelist,
                        WhitelistExtension, normalize_program, parse_segments, validate)


class BashExecutor:
    def __init__(self, whitelist: Whitelist, extension: WhitelistExtension | None = None,
                 cwd: str | None = None):
        self.whitelist = whitelist
        self.extension = extension
        self.cwd = cwd or os.getcwd()

    def run(self, command: str) -> str:
        """执行命令。返回 stdout（成功）或 BlockResult JSON（拦截）。"""
        br = validate(command, self.whitelist, self.extension)  # layer-2 兜底
        if br is not None:
            return br.to_json()
        try:
            segments, ops = parse_segments(command)
        except ValueError as e:
            log_error("parse_segments_failed", command=command, error=repr(e))
            return BlockResult(True, command, Reason.NOT_IN_WHITELIST, BLOCK_MESSAGE).to_json()
        return self._execute(segments, ops)

    def _execute(self, segments: list[list[str]], ops: list[str]) -> str:
        outputs: list[str] = []
        prev_stdout: str | None = None
        for idx, seg in enumerate(segments):
            if not seg:
                continue
            if normalize_program(seg[0]) == "cd":
                self._apply_cd(seg)
                continue
            stdin_data = prev_stdout if (idx > 0 and ops and ops[idx - 1] == "|") else None
            try:
                p = subprocess.run(seg, shell=False, cwd=self.cwd,
                                   input=stdin_data, capture_output=True, text=True)
            except FileNotFoundError as e:
                log_error("command_not_found", program=seg[0], error=repr(e))
                return f"命令未找到：{seg[0]}"
            prev_stdout = p.stdout
            if p.stdout:
                outputs.append(p.stdout)
            if p.returncode != 0 and p.stderr:
                outputs.append(p.stderr)
        return "\n".join(o for o in outputs if o).strip()

    def _apply_cd(self, seg: list[str]) -> None:
        if len(seg) >= 2:
            target = seg[1]
            new = target if os.path.isabs(target) else os.path.join(self.cwd, target)
            self.cwd = os.path.normpath(new)
        else:
            self.cwd = os.path.expanduser("~")
