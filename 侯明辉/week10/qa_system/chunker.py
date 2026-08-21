# week10检索增强生成/qa_system/chunker.py
"""将课程 markdown 按标题语义分块，产出带元数据的块列表。"""
import re
from pathlib import Path

import config

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _week_label(filename: str) -> str:
    """week10.md -> week10；名词解释.md -> 名词解释。"""
    stem = Path(filename).stem
    m = re.match(r"(week\d+)", stem, re.IGNORECASE)
    return m.group(1).lower() if m else stem


def _split_sections(text: str):
    """按 `## ` 行切节，丢弃首个 `## ` 之前的前言以及 `## 目录` 自身。返回 (heading, body) 列表。"""
    lines = text.splitlines()
    sections = []
    heading = None
    buf = []
    for line in lines:
        if line.startswith("## "):
            if heading is not None and heading != "目录":
                sections.append((heading, "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
        elif heading is not None:
            buf.append(line)
    if heading is not None and heading != "目录":
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _doc_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return ""


def chunk_markdown(file_path, max_chunk_size=None, overlap=None) -> list[dict]:
    max_chunk_size = max_chunk_size or config.CHUNK_MAX_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8")

    week = _week_label(file_path.name)
    doc_title = _doc_title(text)
    chunks: list[dict] = []
    step = max(1, max_chunk_size - overlap)

    for sec_idx, (heading, body) in enumerate(_split_sections(text)):
        body = _COMMENT_RE.sub("", body).strip()
        if not body:                      # 清理后为空（如纯 Q&A 占位节）跳过
            continue
        section_path = f"{doc_title} > {heading}" if doc_title else heading

        pieces = []
        if len(body) <= max_chunk_size:
            pieces = [body]
        else:
            start = 0
            while start < len(body):
                pieces.append(body[start:start + max_chunk_size])
                start += step

        for piece_idx, content in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{week}#{sec_idx}#{piece_idx}",
                "source_file": file_path.name,
                "week": week,
                "section_path": section_path,
                "content": content.strip(),
            })
    return chunks