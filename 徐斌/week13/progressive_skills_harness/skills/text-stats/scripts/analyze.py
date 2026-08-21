"""文本统计脚本：输出 JSON 报告。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def analyze(text: str) -> dict:
    if text is None:
        raise ValueError("text is None")
    lines = text.splitlines() or ([""] if text == "" else [])
    words = re.findall(r"[A-Za-z0-9']+", text)
    cjk = CJK_RE.findall(text)
    chars = len(text)
    chars_no_space = len(re.sub(r"\s+", "", text))
    result = {
        "chars": chars,
        "chars_no_space": chars_no_space,
        "words": len(words),
        "cjk_chars": len(cjk),
        "lines": len(lines) if text != "" else 0,
        "summary": (
            f"共 {chars} 字符（去空白 {chars_no_space}），"
            f"{len(words)} 个英文词，{len(cjk)} 个汉字，{len(lines) if text else 0} 行"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze text statistics")
    parser.add_argument("--text", default=None, help="直接传入文本")
    parser.add_argument("--file", default=None, help="从文件读取文本")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(json.dumps({"error": f"file not found: {args.file}"}, ensure_ascii=False))
            return 1
        text = path.read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    if text.strip() == "":
        print(json.dumps({"error": "empty text"}, ensure_ascii=False))
        return 1

    result = analyze(text)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
