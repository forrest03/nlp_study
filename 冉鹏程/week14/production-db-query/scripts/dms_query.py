#!/usr/bin/env python3
"""通过 DMS 执行带 EXPLAIN 审批闸门的生产只读查询。"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

QUERY_URL = "https://dms.yzw.cn/query/"
COOKIE_FILE = Path.home() / "Documents" / ".dms_cookie"
MAX_COOKIE_FILE_BYTES = 20_000
FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|REPLACE|MERGE|ALTER|DROP|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"SET|USE|CALL|LOAD|LOCK|UNLOCK|SLEEP|BENCHMARK|HANDLER)\b|\bINTO\s+OUTFILE\b|\bFOR\s+UPDATE\b",
    re.IGNORECASE,
)


def _remove_comments_and_literals(sql: str) -> str:
    """移除注释和字面量，避免误判其中出现的 SQL 关键字。"""
    result, index, quote = [], 0, None
    while index < len(sql):
        pair = sql[index:index + 2]
        if quote is None and pair == "--":
            index = sql.find("\n", index)
            index = len(sql) if index < 0 else index
        elif quote is None and pair == "/*":
            index = sql.find("*/", index + 2)
            index = len(sql) if index < 0 else index + 2
        elif sql[index] in "'\"`":
            quote = None if quote == sql[index] else (sql[index] if quote is None else quote)
            result.append(" ")
            index += 1
        elif quote is not None:
            result.append(" ")
            index += 1
        else:
            result.append(sql[index])
            index += 1
    return "".join(result)


def _validate_read_only(sql: str) -> None:
    """校验 SQL 是单条、只读且不含明显高风险语法。"""
    inspected = _remove_comments_and_literals(sql).strip()
    if not inspected or len(sql) > 200_000:
        raise ValueError("SQL 不能为空，且长度不能超过 200000 字符。")
    if ";" in inspected or not re.match(r"^(SELECT|WITH)\b", inspected, re.IGNORECASE):
        raise ValueError("仅允许一条不带分号的 SELECT 或 WITH 只读查询。")
    if FORBIDDEN_SQL.search(inspected):
        raise ValueError("SQL 含写操作、锁定读或高风险函数，已拒绝执行。")


def _load_approval(path: Path, sql_digest: str, explain_digest: str) -> None:
    """校验审批绑定当前 SQL 和当前 EXPLAIN 响应，防止绕过性能审阅。"""
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取审批文件：{error}") from error
    if approval.get("verdict") != "approved" or approval.get("sql_sha256") != sql_digest:
        raise ValueError("审批文件未放行当前 SQL；请重新审阅 EXPLAIN。")
    if approval.get("explain_response_sha256") != explain_digest:
        raise ValueError("当前 EXPLAIN 与审批结果不一致；请重新审阅后再执行。")


def _get_credentials() -> tuple[str, str]:
    """从受限本地 Cookie 文件取得临时凭据，禁止回退到其他来源。"""
    cookie = _read_cookie_file()
    csrf_token = _get_csrf_from_cookie(cookie)
    if not csrf_token:
        raise ValueError("Cookie 文件未包含 csrftoken。")
    return cookie, csrf_token


def _get_csrf_from_cookie(cookie: str) -> str:
    """从环境变量中的 Cookie 临时提取 csrftoken，不保存或打印凭据。"""
    for item in cookie.split(";"):
        name, separator, value = item.strip().partition("=")
        if separator and name.lower() == "csrftoken":
            return value
    return ""


def _read_cookie_file() -> str:
    """读取并校验当前用户的 Cookie 文件，不记录其内容。"""
    try:
        if COOKIE_FILE.is_symlink() or not COOKIE_FILE.is_file():
            raise ValueError("Cookie 文件必须是普通文件。")
        file_status = COOKIE_FILE.stat()
        if file_status.st_size <= 0 or file_status.st_size > MAX_COOKIE_FILE_BYTES:
            raise ValueError("Cookie 文件大小不合法。")
        if os.name != "nt" and file_status.st_mode & 0o077:
            raise ValueError("Cookie 文件权限必须为 600。")
        cookie = COOKIE_FILE.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(f"无法读取 Cookie 文件：{error}") from error
    if not cookie or "\n" in cookie or "\r" in cookie:
        raise ValueError("Cookie 文件内容不合法。")
    return cookie


def _get_explain_digest(response: bytes) -> str:
    """计算稳定的执行计划摘要，排除 query_time 等每次变化的响应字段。"""
    try:
        data = json.loads(response.decode("utf-8")).get("data", {})
        plan = {"column_list": data["column_list"], "rows": data["rows"]}
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
        raise ValueError(f"无法解析 EXPLAIN 执行计划：{error}") from error
    normalized = json.dumps(plan, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _post_query(sql: str, arguments: argparse.Namespace) -> bytes:
    """调用 DMS 查询接口；网络错误显式失败且不输出认证信息。"""
    cookie, csrf_token = _get_credentials()
    form = {"instance_name": arguments.instance, "db_name": arguments.database,
            "schema_name": "", "tb_name": "", "sql_content": sql,
            "limit_num": str(arguments.limit)}
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
               "Cookie": cookie, "X-CSRFToken": csrf_token, "X-Requested-With": "XMLHttpRequest"}
    request = Request(QUERY_URL, data=urlencode(form).encode(), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=arguments.timeout) as response:
            return response.read()
    except HTTPError as error:
        raise RuntimeError(f"DMS 请求失败，HTTP 状态码：{error.code}") from error
    except URLError as error:
        raise RuntimeError(f"DMS 网络请求失败：{error.reason}") from error


def _parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数，明确区分 EXPLAIN 与正式查询阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("explain", "query"), required=True)
    parser.add_argument("--sql-file", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instance", default="tidb-hw-01.yzw.cn-45-100")
    parser.add_argument("--database", choices=("pbets", "pbid"), default="pbets")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args(argv)


def main() -> int:
    """执行指定阶段；返回 0 表示成功，非 0 表示未执行或请求失败。"""
    try:
        arguments = _parse_arguments()
        if not 1 <= arguments.limit <= 10000 or not 1 <= arguments.timeout <= 120:
            raise ValueError("limit 必须在 1-10000，timeout 必须在 1-120。")
        sql = arguments.sql_file.read_text(encoding="utf-8").strip()
        _validate_read_only(sql)
        digest = hashlib.sha256(sql.encode()).hexdigest()
        explain_response = _post_query(f"EXPLAIN {sql}", arguments)
        explain_digest = _get_explain_digest(explain_response)
        if arguments.phase == "explain":
            response = explain_response
        else:
            if arguments.approval_file is None:
                raise ValueError("正式查询必须提供 --approval-file。")
            _load_approval(arguments.approval_file, digest, explain_digest)
            response = _post_query(sql, arguments)
        arguments.output.write_bytes(response)
        print(f"阶段完成：{arguments.phase}；SQL SHA-256：{digest}；EXPLAIN SHA-256：{explain_digest}；响应已写入指定文件。")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"未执行或执行失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
