#!/usr/bin/env python3
"""Smoke-test the remote embed/rerank service."""

import json
import sys
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request


BASE_URL = "https://open-inner.yzwqa.cn/api"
APP_KEY = "WL2y70uf"
VERSION = "1.0"

METHOD_HEALTH = "embed_rerank_server.health"
METHOD_EMBEDDINGS = "embed_rerank_server.v1.embeddings"
METHOD_RERANK = "embed_rerank_server.v1.rerank"

DEFAULT_TIMEOUT_SECONDS = 30


def build_url(method_name: str) -> str:
    """Build a full request URL with fixed gateway parameters."""
    query = urllib.parse.urlencode(
        {
            "appkey": APP_KEY,
            "version": VERSION,
            "method": method_name,
        }
    )
    return f"{BASE_URL}?{query}"


def get_json(method_name: str) -> Dict[str, Any]:
    """Send one GET request and return the decoded JSON response."""
    request = urllib.request.Request(
        url=build_url(method_name),
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        content = response.read().decode("utf-8")
    return json.loads(content)


def post_json(
    method_name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send one JSON request and return the decoded JSON response."""
    body = b""
    headers = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=build_url(method_name),
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        content = response.read().decode("utf-8")
    return json.loads(content)


def run_health_check() -> None:
    """Call the health endpoint and print the response."""
    print("== health ==")
    print_json(get_json(METHOD_HEALTH))


def run_embeddings_test() -> None:
    """Call the embeddings endpoint and print the response."""
    payload = {
        "input": [
            "第一段测试文本",
            "第二段测试文本",
        ]
    }
    print("== embeddings ==")
    print_json(post_json(METHOD_EMBEDDINGS, payload))


def run_rerank_test() -> None:
    """Call the rerank endpoint and print the response."""
    payload = {
        "query": "投标文件是否满足资格要求",
        "documents": [
            "文档A包含资格要求说明",
            "文档B主要描述付款方式",
            "文档C列出了投标人资质条件",
        ],
    }
    print("== rerank ==")
    print_json(post_json(METHOD_RERANK, payload))


def print_json(payload: Dict[str, Any]) -> None:
    """Pretty-print one JSON payload."""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: List[str]) -> int:
    """Run selected smoke tests from the command line."""
    actions = {
        "health": run_health_check,
        "embeddings": run_embeddings_test,
        "rerank": run_rerank_test,
        "all": run_health_check,
    }
    selected = argv[1] if len(argv) > 1 else "all"
    try:
        if selected == "all":
            run_health_check()
            run_embeddings_test()
            run_rerank_test()
            return 0
        if selected not in actions:
            print("usage: python scripts/test_remote_api.py [all|health|embeddings|rerank]")
            return 1
        actions[selected]()
        return 0
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"http error: status={exc.code} body={error_body}", file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        print(f"url error: reason={exc.reason}", file=sys.stderr)
        return 3
    except json.JSONDecodeError as exc:
        print(f"invalid json response: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
