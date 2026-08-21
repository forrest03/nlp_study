#!/usr/bin/env python3
"""校验当前用户 Documents 目录中的 DMS Cookie 文件。"""

import sys

from dms_query import _get_credentials

def main() -> int:
    """校验当前环境或 macOS 钥匙串中的会话；返回 0 表示可安全使用。"""
    try:
        _get_credentials()
    except ValueError:
        print("未检测到有效 DMS 会话。", file=sys.stderr)
        return 1
    print("DMS 临时会话已就绪；Cookie 未写入磁盘或输出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
