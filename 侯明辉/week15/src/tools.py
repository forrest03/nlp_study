"""
本地文档 I/O 工具：read_file + list_files。

read_file 输入格式: "path:start_line:end_line"
  例: "week15.md:1:50"  → 读 week15.md 第 1~50 行
  例: "week15.md"        → 读全部（默认 0~500 行）

list_files 输入格式: "."  → 列出 corpus_dir 顶层目录
                "week10"  → 列出 corpus_dir/week10/ 下所有文件
                "" 或 "*"  → 同 "."

路径越界防护：resolve 后 startswith 检查，禁止 ../ 跳出 corpus_dir。

依赖：pathlib
"""
from pathlib import Path
import os

# 默认 corpus_dir：week15homework_agent 的父目录的父目录
# 解析: src/tools.py -> src/ -> week15homework_agent/ -> week15graph和llm/ -> 仓库根
DEFAULT_CORPUS_DIR = Path(__file__).parent.parent.parent.parent
MAX_LINES = 500


def _resolve_safe(rel_path: str, corpus_dir: Path) -> tuple[Path | None, str | None]:
    """
    把相对路径解析为绝对路径，并检查是否在 corpus_dir 内。
    返回 (full_path, error_msg)。成功时 error_msg 为 None。
    """
    # 空路径 → 视为 corpus_dir
    if not rel_path or rel_path == ".":
        return corpus_dir, None
    try:
        full = (corpus_dir / rel_path).resolve()
    except Exception as e:
        return None, f"路径解析失败: {e}"
    corpus_resolved = corpus_dir.resolve()
    full_s = str(full)
    root_s = str(corpus_resolved)
    if not (full_s == root_s or full_s.startswith(root_s + os.sep)):
        return None, f"错误：禁止访问 corpus_dir 之外的文件（{rel_path}）"
    return full, None


def read_file(action_input: str, corpus_dir: Path | None = None) -> str:
    """
    读文件片段。action_input 格式: "path:start_line:end_line"
    """
    corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
    parts = action_input.strip().rsplit(":", 2)
    rel_path = parts[0]
    start = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    end = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else start + MAX_LINES

    full, err = _resolve_safe(rel_path, corpus_dir)
    if err:
        return err
    if not full.exists():
        return f"错误：文件不存在 {rel_path}"
    if not full.is_file():
        return f"错误：{rel_path} 不是文件"

    try:
        lines = full.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        try:
            lines = full.read_text(encoding="gbk").splitlines()
        except Exception as e:
            return f"错误：读取失败 {type(e).__name__}: {str(e)[:80]}"

    end = min(end, len(lines))
    start = max(0, min(start, end))
    selected = lines[start:end]
    header = f"[{rel_path} 第 {start+1}~{end} 行 / 共 {len(lines)} 行]\n"
    return header + "\n".join(selected)


def list_files(action_input: str, corpus_dir: Path | None = None) -> str:
    """
    列文件。action_input: "." / "" / "week10" / "*.md"
    """
    corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
    rel = action_input.strip().strip('"') or "."
    full, err = _resolve_safe(rel, corpus_dir)
    if err:
        return err
    if not full.exists():
        return f"错误：路径不存在 {rel}"

    if full.is_file():
        return f"[文件] {full.relative_to(corpus_dir.resolve())}"
    if not full.is_dir():
        return f"错误：{rel} 不是文件或目录"

    # 列目录下文件 + 子目录
    lines = []
    try:
        for child in sorted(full.iterdir()):
            if child.name.startswith("."):
                continue
            tag = "/" if child.is_dir() else ""
            lines.append(f"  {child.name}{tag}")
    except PermissionError as e:
        return f"错误：权限不足 {e}"
    rel_str = str(full.relative_to(corpus_dir.resolve())) or "."
    header = f"[{rel_str}/ 目录列表]\n"
    return header + "\n".join(lines) if lines else header + "（空目录）"


if __name__ == "__main__":
    # 自测：读 week15.md 头 20 行 + 列 week15graph和llm/ 目录
    import sys
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    # Windows console (GBK) 编码下打印中文/特殊符号需要 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=== 测试 1: read_file week15.md 头 20 行 ===")
    out = read_file("week15graph和llm/week15.md:0:20")
    print(out[:500])
    assert "[week15graph和llm/week15.md 第 1~" in out, "格式头缺失"
    assert "# Week 15" in out, "未读到预期内容"

    print("\n=== 测试 2: list_files week15graph和llm/ ===")
    out = list_files("week15graph和llm")
    print(out)
    assert "week15.md" in out, "未列出 week15.md"

    print("\n=== 测试 3: 路径越界防护 ===")
    out = read_file("../../etc/passwd:0:10")
    print(out)
    assert "禁止访问" in out, "路径越界未拦截"

    print("\n=== 测试 4: 文件不存在 ===")
    out = read_file("nonexistent_xyz.md")
    print(out)
    assert "文件不存在" in out

    print("\n✓ tools.py 自测全部通过")
