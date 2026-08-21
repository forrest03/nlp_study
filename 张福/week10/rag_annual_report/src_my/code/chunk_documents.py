"""
文档分块脚本：对解析后的 JSON 文件做分块处理

三种分块策略：
  ┌──────────┬─────────────┬───────────┬───────────┐
  │ 策略     │ 切割依据    │ chunk大小 │ 适合场景  │
  ├──────────┼─────────────┼───────────┼───────────┤
  │ fixed    │ 每500字符截断│ 均匀      │ Baseline  │
  │ semantic │ 遇标题强制切│ 不均匀    │ 默认推荐  │
  │ hierarchical │ 父子双层 │ 双层      │ 长文档精确│
  └──────────┴─────────────┴───────────┴───────────┘

输出格式：
  {
    "chunk_id": "doc_name_00001",
    "content": "报告期内，公司实现营业总收入...",
    "metadata": {
      "page_num": 56,
      "section": "第十节 > 二、财务报表 > 利润表",
      "block_types": ["text"],
      "is_ocr": false,
      "strategy": "semantic",
      "source_file": "xxx.pdf"
    }
  }

依赖包：
  json (内置)
  uuid (内置)
  logging (内置)
  pathlib (内置)
  typing (内置)

运行方式：
  conda activate py312
  python src_my/code/chunk_documents.py
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "resource" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "resource" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "semantic"


def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    """
    按字符数切块，相邻块有重叠。
    缺点：无视句子/段落边界，表格会被切断。
    优点：实现最简单，块大小可预测。
    """
    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 800,
    min_chunk_size: int = 20,
) -> Iterator[dict]:
    """
    按解析结构分块：遇到标题强制切块，段落尽量合并到 max_chunk_size 以内。
    优点：保留语义完整性，章节边界清晰。
    缺点：块大小不均匀（财务报表单个表格可能很大）。
    """
    buffer_blocks = []
    buffer_len = 0

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        meta = {
            "page_num": buf[0]["page_num"],
            "section": " > ".join(buf[0]["section_path"]) if buf[0]["section_path"] else "",
            "block_types": list({b["block_type"] for b in buf}),
            "is_ocr": any(b["is_ocr"] for b in buf),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen = len(block["content"])

        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len = 0

            if blen >= min_chunk_size:
                yield {
                    "content": block["content"],
                    "metadata": {
                        "page_num": block["page_num"],
                        "section": " > ".join(block["section_path"]) if block["section_path"] else "",
                        "block_types": ["title"],
                        "is_ocr": block["is_ocr"],
                    }
                }
            continue

        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len = 0
            yield {
                "content": block["content"],
                "metadata": {
                    "page_num": block["page_num"],
                    "section": " > ".join(block["section_path"]) if block["section_path"] else "",
                    "block_types": ["table"],
                    "is_ocr": block["is_ocr"],
                }
            }
            continue

        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len = 0

        buffer_blocks.append(block)
        buffer_len += blen

    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result


def chunk_hierarchical(
    blocks: list[dict],
    parent_size: int = 2000,
    child_size: int = 400,
    overlap: int = 50,
) -> Iterator[dict]:
    """
    两级结构：父块（大段落，用于给 LLM 提供足够上下文）+ 子块（小段落，用于向量检索）
    检索时：命中子块 → 取父块内容 → 给 LLM 读父块（Small-to-Big Retrieval）
    """
    full_text = "\n\n".join(b["content"] for b in blocks if b["content"].strip())

    parents = []
    start = 0
    while start < len(full_text):
        end = min(start + parent_size, len(full_text))
        content = full_text[start:end]
        parent_id = str(uuid.uuid4())[:8]
        parents.append({
            "parent_id": parent_id,
            "content": content,
            "start": start,
            "end": end,
        })
        start += parent_size - overlap

    for parent in parents:
        p_content = parent["content"]
        p_id = parent["parent_id"]
        c_start = 0
        while c_start < len(p_content):
            c_end = min(c_start + child_size, len(p_content))
            child_content = p_content[c_start:c_end]
            yield {
                "content": child_content,
                "metadata": {
                    "parent_id": p_id,
                    "parent_content": p_content,
                    "block_types": ["text"],
                    "is_ocr": False,
                    "section": "",
                    "page_num": -1,
                }
            }
            c_start += child_size - overlap


def build_chunk_id(source_name: str, idx: int) -> str:
    name_part = source_name.replace(".json", "")[:20]
    name_part = name_part.replace(" ", "_").replace("《", "").replace("》", "")
    return f"{name_part}_{idx:05d}"


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    blocks = data.get("blocks", [])

    source_file = meta.get("filename", parsed_path.name.replace(".json", ".pdf"))

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    raw_chunks = []

    if strategy == "fixed":
        full_text = "\n\n".join(b["content"] for b in blocks)
        for text_chunk in chunk_fixed(full_text):
            raw_chunks.append({
                "content": text_chunk,
                "metadata": {"block_types": ["text"], "is_ocr": False, "section": "", "page_num": -1}
            })

    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)

    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)

    else:
        raise ValueError(f"未知策略: {strategy}")

    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(parsed_path.stem, idx)
        chunk["chunk_id"] = chunk_id
        chunk["metadata"]["strategy"] = strategy
        chunk["metadata"]["source_file"] = source_file
        result.append(chunk)

    out_path = CHUNKS_DIR / f"{parsed_path.stem}_{strategy}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"  → {len(result)} 个 chunk，已保存 {out_path.name}")
    return result


def main():
    parsed_files = list(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        logger.error("没有找到解析结果，请先运行 parse_pdf.py")
        return

    all_chunks = []
    for path in parsed_files:
        chunks = process_file(path, strategy=STRATEGY)
        all_chunks.extend(chunks)

    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    table_count = sum(1 for c in all_chunks if "table" in c["metadata"].get("block_types", []))
    ocr_count = sum(1 for c in all_chunks if c["metadata"].get("is_ocr"))
    logger.info(f"其中表格块: {table_count}  OCR块: {ocr_count}")


if __name__ == "__main__":
    main()