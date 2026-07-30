"""whitelist.py - 纯函数白名单校验：解析 / 归一化 / 拆分 / 严格拦截 / 白名单匹配。

确定性校验，不委托模型（规则 5）。Agent 层与 Bash 工具层共用同一 validate（FR-006 双层）。
拦截项与 reason 映射见 contracts/block-result.md。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .logging import log_error


class Reason(str, Enum):
    NOT_IN_WHITELIST = "NOT_IN_WHITELIST"
    COMPOUND_NONCOMPLIANT = "COMPOUND_NONCOMPLIANT"
    COMMAND_SUBSTITUTION = "COMMAND_SUBSTITUTION"
    INTERPRETER_PASSTHROUGH = "INTERPRETER_PASSTHROUGH"
    INDIRECT_DELETE = "INDIRECT_DELETE"
    REDIRECT_OVERWRITE = "REDIRECT_OVERWRITE"
    FIND_FORBIDDEN_FLAG = "FIND_FORBIDDEN_FLAG"


BLOCK_MESSAGE = "该操作不被允许，请调整方案后重试"

# 默认白名单（spec 配置项）。cd 由 BashExecutor 特殊处理（更新 cwd，不 spawn）。
DEFAULT_COMMANDS = {
    "ls", "pwd", "cat", "echo", "mkdir", "cd", "head", "tail",
    "wc", "grep", "find", "stat", "whoami", "date",
}
# 每条命令的禁用 flag（spec: find 禁 -delete / -exec）
DEFAULT_CONSTRAINTS = {"find": {"-delete", "-exec"}}

# 危险解释器透传（FR-004）：sh -c / bash -c / eval / xargs 等，整体拦截
INTERPRETERS = {"sh", "bash", "dash", "zsh", "fish", "csh", "tcsh", "eval", "xargs"}

# 重定向操作符检测（FR-004 + F5 扩展：含 < / 2> / &> 等，deny-by-default）
_REDIRECT_RE = re.compile(r"^\d*&?[<>]")


@dataclass
class Whitelist:
    commands: set[str] = field(default_factory=set)
    constraints: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Whitelist":
        return cls(
            commands=set(DEFAULT_COMMANDS),
            constraints={k: set(v) for k, v in DEFAULT_CONSTRAINTS.items()},
        )


@dataclass
class WhitelistExtension:
    """临时扩展（FR-014），绑定激活 Skill。validate 用 base ∪ extension。"""
    commands: set[str] = field(default_factory=set)
    bound_skill: str = ""


@dataclass
class BlockResult:
    blocked: bool
    command: str
    reason: Reason
    message: str

    def to_json(self) -> str:
        import json
        return json.dumps({
            "blocked": self.blocked,
            "command": self.command,
            "reason": self.reason.value,
            "message": self.message,
        }, ensure_ascii=False)


def normalize_program(program: str) -> str:
    """归一化程序名（FR-002）：/bin/ls、./ls、ls -> ls。"""
    return program.rsplit("/", 1)[-1]


def _has_redirect(tokens: list[str]) -> bool:
    return any(_REDIRECT_RE.match(t) for t in tokens)


def _tokenize(command: str) -> list[str]:
    """quote-aware 分词：仅以空白与操作符 ``;`` ``&`` ``|`` 切分，引号内不切分（posix 去引号）。

    非 ASCII 字符（如 CJK 路径 ``张雷``）随所属 token 整体保留，不被拆成单字--
    ``shlex(punctuation_chars=...)`` 会把 CJK 拆成单字 token，破坏含非 ASCII 的路径
    （T029）。``;`` ``&`` ``|`` ``&&`` ``||`` 作为独立操作符 token 返回。
    """
    tokens: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote == "'":  # posix 单引号：内容字面，不转义
            if c == "'":
                quote = None
            else:
                cur.append(c)
            i += 1
            continue
        if quote == '"':  # posix 双引号：仅 \ 转义下一字符
            if c == '"':
                quote = None
            elif c == "\\" and i + 1 < n:
                cur.append(command[i + 1])
                i += 2
                continue
            else:
                cur.append(c)
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue
        if c in ";&|":
            if cur:
                tokens.append("".join(cur))
                cur = []
            if c == "&" and i + 1 < n and command[i + 1] == "&":
                tokens.append("&&")
                i += 2
                continue
            if c == "|" and i + 1 < n and command[i + 1] == "|":
                tokens.append("||")
                i += 2
                continue
            tokens.append(c)
            i += 1
            continue
        if c.isspace():
            if cur:
                tokens.append("".join(cur))
                cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if cur:
        tokens.append("".join(cur))
    return tokens


def parse_segments(command: str) -> tuple[list[list[str]], list[str]]:
    """拆分组合命令（FR-003）。

    返回 (segments, ops)：segments 为各子命令的 token 列表；ops 为子命令间的操作符
    （'|' 表示管道，';' 表示顺序，含 ; && || &）。quote-aware（引号内的 ; 等不拆分）。
    """
    tokens = _tokenize(command.strip())
    _DELIMITERS = {";", "&", "|", "&&", "||"}
    segments: list[list[str]] = [[]]
    ops: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t in _DELIMITERS:
            is_pipe = (t == "|")
            j = i
            while j < n and tokens[j] in _DELIMITERS:
                if tokens[j] == "|":
                    is_pipe = True
                j += 1
            ops.append("|" if is_pipe else ";")
            segments.append([])
            i = j
        else:
            segments[-1].append(t)
            i += 1
    return segments, ops


def _check_segment(tokens: list[str], whitelist: Whitelist,
                   extension: WhitelistExtension | None) -> Reason | None:
    """校验单个子命令。返回 None 表示放行，返回 Reason 表示命中拦截。"""
    program = normalize_program(tokens[0])
    if program in INTERPRETERS:
        return Reason.INTERPRETER_PASSTHROUGH
    if program == "truncate":
        return Reason.INDIRECT_DELETE
    forbidden = whitelist.constraints.get(program)
    if forbidden and any(t in forbidden for t in tokens):
        return Reason.FIND_FORBIDDEN_FLAG  # find -delete / -exec（F1：归此 reason）
    allowed = whitelist.commands
    if extension is not None:
        allowed = allowed | extension.commands
    if program not in allowed:
        return Reason.NOT_IN_WHITELIST
    return None


def validate(command: str, whitelist: Whitelist,
             extension: WhitelistExtension | None = None) -> BlockResult | None:
    """主校验入口。返回 None = 放行；返回 BlockResult = 拦截。"""
    raw = (command or "").strip()
    if not raw:
        return None
    # 1. 命令替换：$(...) 或反引号，任一出现即拦截（含引号内）
    if "$(" in raw or "`" in raw:
        return BlockResult(True, command, Reason.COMMAND_SUBSTITUTION, BLOCK_MESSAGE)
    # 2. 解析拆分（quote-aware）
    try:
        segments, _ = parse_segments(raw)
    except ValueError as e:
        log_error("parse_segments_failed", command=command, error=repr(e))
        return BlockResult(True, command, Reason.NOT_IN_WHITELIST, BLOCK_MESSAGE)
    # 3. 重定向：token 扫描
    all_tokens = [t for seg in segments for t in seg]
    if _has_redirect(all_tokens):
        return BlockResult(True, command, Reason.REDIRECT_OVERWRITE, BLOCK_MESSAGE)
    # 4. 逐子命令校验；任一不合规，组合命令整体拦截（FR-003）
    non_empty = [s for s in segments if s]
    compound = len(non_empty) > 1
    for seg in non_empty:
        r = _check_segment(seg, whitelist, extension)
        if r is not None:
            reason = Reason.COMPOUND_NONCOMPLIANT if compound else r
            return BlockResult(True, command, reason, BLOCK_MESSAGE)
    return None
