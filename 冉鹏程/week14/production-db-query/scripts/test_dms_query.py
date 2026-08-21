"""验证生产查询 SQL 的本地安全闸门。"""

import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import run
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("dms_query.py")
SPEC = importlib.util.spec_from_file_location("dms_query", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReadOnlySqlValidationTest(unittest.TestCase):
    """验证只读 SQL 校验能阻止写入和危险语法。"""

    def test_accepts_select_with_literal_keyword(self):
        """验证字面量中的关键字不会造成误拒绝。"""
        MODULE._validate_read_only("SELECT 'delete' AS label FROM bet_evaluation_result")

    def test_rejects_write_statement(self):
        """验证写操作会在网络请求前被拒绝。"""
        with self.assertRaises(ValueError):
            MODULE._validate_read_only("DELETE FROM bet_evaluation_result")

    def test_rejects_multiple_statements(self):
        """验证多语句不会穿透只读闸门。"""
        with self.assertRaises(ValueError):
            MODULE._validate_read_only("SELECT 1; SELECT 2")

    def test_rejects_locking_read(self):
        """验证锁定读不会在生产库上执行。"""
        with self.assertRaises(ValueError):
            MODULE._validate_read_only("SELECT * FROM bet_evaluation_result FOR UPDATE")

    def test_limits_database_to_supported_scope(self):
        """验证命令行仅接受 PBETS 和 PBID 两个生产库。"""
        arguments = MODULE._parse_arguments([
            "--phase", "explain", "--database", "pbid", "--sql-file", "query.sql", "--output", "plan.json",
        ])
        self.assertEqual("pbid", arguments.database)
        with self.assertRaises(SystemExit):
            MODULE._parse_arguments([
                "--phase", "explain", "--database", "other", "--sql-file", "query.sql", "--output", "plan.json",
            ])

    def test_approval_must_match_sql_and_explain_response(self):
        """验证审批必须绑定 SQL 和已审阅的 EXPLAIN 响应。"""
        with tempfile.TemporaryDirectory() as directory:
            approval_path = Path(directory) / "approval.json"
            approval_path.write_text(json.dumps({
                "verdict": "approved", "sql_sha256": "sql", "explain_response_sha256": "plan",
            }), encoding="utf-8")
            MODULE._load_approval(approval_path, "sql", "plan")
            with self.assertRaises(ValueError):
                MODULE._load_approval(approval_path, "sql", "changed-plan")

    def test_query_replays_explain_before_formal_sql(self):
        """验证正式 SQL 只能在当前 EXPLAIN 与审批匹配后发出。"""
        with tempfile.TemporaryDirectory() as directory:
            sql_path, approval_path = Path(directory) / "query.sql", Path(directory) / "approval.json"
            output_path, plan, sql = Path(directory) / "result.json", b'{"data":{"column_list":["id"],"rows":[["IndexRangeScan"]]}}', "SELECT 1"
            sql_path.write_text(sql, encoding="utf-8")
            approval_path.write_text(json.dumps({"verdict": "approved",
                "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                "explain_response_sha256": MODULE._get_explain_digest(plan)}), encoding="utf-8")
            calls = []
            with patch.object(MODULE, "_post_query", side_effect=lambda value, _: calls.append(value) or plan):
                with patch.object(sys, "argv", ["dms_query.py", "--phase", "query", "--sql-file", str(sql_path),
                                                  "--approval-file", str(approval_path), "--output", str(output_path)]):
                    self.assertEqual(0, MODULE.main())
            self.assertEqual(["EXPLAIN SELECT 1", "SELECT 1"], calls)

    def test_uses_restricted_cookie_file(self):
        """验证凭据仅从权限受限的 Cookie 文件读取。"""
        with tempfile.TemporaryDirectory() as directory:
            cookie_path = Path(directory) / ".dms_cookie"
            cookie_path.write_text("session=temporary; csrftoken=csrf-file", encoding="utf-8")
            os.chmod(cookie_path, 0o600)
            with patch.object(MODULE, "COOKIE_FILE", cookie_path):
                self.assertEqual(("session=temporary; csrftoken=csrf-file", "csrf-file"), MODULE._get_credentials())

    def test_rejects_permissive_cookie_file(self):
        """验证非 Windows 环境拒绝其他用户可读取的 Cookie 文件。"""
        if os.name == "nt":
            self.skipTest("Windows 使用 ACL 管理文件权限。")
        with tempfile.TemporaryDirectory() as directory:
            cookie_path = Path(directory) / ".dms_cookie"
            cookie_path.write_text("session=temporary; csrftoken=csrf-file", encoding="utf-8")
            os.chmod(cookie_path, 0o644)
            with patch.object(MODULE, "COOKIE_FILE", cookie_path):
                with self.assertRaises(ValueError):
                    MODULE._get_credentials()

    def test_explain_digest_ignores_dynamic_response_fields(self):
        """验证审批摘要仅依赖执行计划，而非查询耗时等动态字段。"""
        first = json.dumps({"data": {"column_list": ["id"], "rows": [["IndexRangeScan"]], "query_time": 0.01}}).encode()
        second = json.dumps({"data": {"column_list": ["id"], "rows": [["IndexRangeScan"]], "query_time": 0.02}}).encode()
        changed = json.dumps({"data": {"column_list": ["id"], "rows": [["TableFullScan"]]}}).encode()
        self.assertEqual(MODULE._get_explain_digest(first), MODULE._get_explain_digest(second))
        self.assertNotEqual(MODULE._get_explain_digest(first), MODULE._get_explain_digest(changed))

    def test_session_check_is_cross_platform_python(self):
        """验证会话校验脚本只依赖 Python 和当前用户 Documents 文件。"""
        session_script = MODULE_PATH.with_name("dms_session.py")
        with tempfile.TemporaryDirectory() as home_directory:
            cookie_directory = Path(home_directory) / "Documents"
            cookie_directory.mkdir()
            cookie_path = cookie_directory / ".dms_cookie"
            cookie_path.write_text("session=temporary; csrftoken=csrf-temporary", encoding="utf-8")
            os.chmod(cookie_path, 0o600)
            environment = {"HOME": home_directory}
            result = run([sys.executable, str(session_script)], env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(0, result.returncode)
            self.assertNotIn("session=temporary", result.stdout)


if __name__ == "__main__":
    unittest.main()
