"""
文档分块脚本：对解析后的年报做分块处理
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ── 策略 A：固定大小分块 ──────────────────────────────────────────────────────

def chunk_fixed(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> Iterator[str]:
    """
    按字符数切块，相邻块有重叠。
    缺点：无视句子/段落边界，表格会被切断。
    优点：实现最简单，块大小可预测。
    """
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        yield {
            "content": text[start:end],
            "start_pos": start,
            "end_pos": end
        }

        if end >= text_len:
            break
        start += chunk_size - overlap


# ── 策略 B：语义分块 ──────────────────────────────────────────────────────────

def chunk_semantic(
    blocks: list[dict],
    max_chunk_size: int = 500,
    min_chunk_size: int = 20,
) -> Iterator[dict]:
    """
    按解析结构分块：遇到标题强制切块，段落尽量合并到 max_chunk_size 以内。

    优点：保留语义完整性，章节边界清晰。
    缺点：块大小不均匀（财务报表单个表格可能很大）。
    """
    buffer_blocks = []
    buffer_len    = 0
    buffer_meta   = {}

    def flush(buf: list[dict]) -> dict | None:
        if not buf:
            return None
        content = "\n\n".join(b["content"] for b in buf)
        # 元信息取第一个块的
        meta = {
            "page_num":   buf[0]["page_num"],
            "section":    " > ".join(buf[0]["section_path"]) if buf[0]["section_path"] else "",
            "block_types": list({b["block_type"] for b in buf}),
            "is_ocr":     any(b["is_ocr"] for b in buf),
        }
        return {"content": content, "metadata": meta}

    for block in blocks:
        btype = block["block_type"]
        blen  = len(block["content"])

        # 标题块：强制先 flush，标题单独作为一个小块（不用于 embedding，
        # 但保留在 chunk 里作为上下文前缀）
        if btype == "title":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0

        # 表格块：单独成块（不与文字合并，防止 embedding 效果变差）
        if btype == "table":
            if buffer_blocks:
                result = flush(buffer_blocks)
                if result and len(result["content"]) >= min_chunk_size:
                    yield result
                buffer_blocks = []
                buffer_len    = 0
            yield {
                "content": block["content"],
                "metadata": {
                    "page_num": block["page_num"],
                    "section":  " > ".join(block["section_path"]),
                    "block_types": ["table"],
                    "is_ocr":   block["is_ocr"],
                }
            }
            continue

        # 文字块：累积，超过 max_chunk_size 就 flush
        if buffer_len + blen > max_chunk_size and buffer_blocks:
            result = flush(buffer_blocks)
            if result and len(result["content"]) >= min_chunk_size:
                yield result
            buffer_blocks = []
            buffer_len    = 0

        buffer_blocks.append(block)
        buffer_len += blen

    # 尾部剩余
    if buffer_blocks:
        result = flush(buffer_blocks)
        if result and len(result["content"]) >= min_chunk_size:
            yield result


# ── 策略 C：层级分块（父子块） ────────────────────────────────────────────────

def chunk_hierarchical(
        blocks: list[dict],
        parent_size: int = 800,
        child_size: int = 200,
        overlap: int = 50,
) -> Iterator[dict]:
    """
    两级结构：
      父块（parent）：大段落，用于给 LLM 提供足够上下文
      子块（child）：小段落，用于向量检索（更精确）

    检索时：命中子块 → 取父块 id → 给 LLM 读父块
    这就是所谓的 "小to大检索"（Small-to-Big Retrieval）

    输出的每个 child 块里带 parent_id 字段。
    """
    import uuid

    # 构建内容到元信息的映射（按内容精确匹配）
    meta_map = {}
    for b in blocks:
        content = b.get("content", "").strip()
        if content:
            meta_map[content] = b

    full_text = "\n\n".join(b["content"] for b in blocks if b["content"].strip())

    if not full_text:
        return

    # 先切父块
    parents = []
    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = min(start + parent_size, text_len)
        content = full_text[start:end]
        parent_id = str(uuid.uuid4())[:8]
        parents.append({
            "parent_id": parent_id,
            "content": content,
            "start": start,
            "end": end,
        })
        if end >= text_len:
            break
        start += parent_size - overlap

    # 再从父块里切子块
    for parent in parents:
        p_content = parent["content"]
        p_id = parent["parent_id"]

        # 从父块内容中提取元信息
        # 找到父块包含的第一个 block
        first_block = None
        for content, meta in meta_map.items():
            if content in p_content:
                first_block = meta
                break

        # 构建章节路径
        section_path = first_block.get("section_path", []) if first_block else []
        section_str = " > ".join(section_path) if section_path else ""

        # 收集父块覆盖的所有页码
        page_nums = set()
        for content, meta in meta_map.items():
            if content in p_content:
                page_num = meta.get("page_num", -1)
                if page_num > 0:
                    page_nums.add(page_num)

        page_num = min(page_nums) if page_nums else -1

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
                    "section": section_str,
                    "page_num": page_num,
                }
            }
            if c_end >= len(p_content):
                break
            c_start += child_size - overlap


# ── 主流程 ────────────────────────────────────────────────────────────────────

STRATEGY = "hierarchical"   # 改成 "fixed" 或 "hierarchical" 体验不同策略

def build_chunk_id(raw_name: str, idx: int) -> str:
    return f"{raw_name}_{idx:05d}"


def process_file(parsed_path: Path, strategy: str = STRATEGY):
    with open(parsed_path, encoding="utf-8") as f:
        data = json.load(f)

    meta   = data.get("meta", {})
    blocks = data.get("blocks", [])

    raw_name = meta.get("raw_name", "unknown")

    logger.info(f"分块 {parsed_path.name}  策略={strategy}  blocks={len(blocks)}")

    # 根据策略生成 chunks
    raw_chunks = []

    if strategy == "fixed":
        for block in blocks:
            for text_chunk in chunk_fixed(block["content"]):
                # 将 section_path 列表拼接成字符串
                section_path = block.get("section_path", [])
                section_str = " > ".join(section_path) if section_path else ""

                raw_chunks.append({
                    "content": text_chunk["content"],
                    "metadata": {
                        "page_num": block.get("page_num", -1),
                        "section": section_str,  # ✅ 现在是 "第一编　总　　则 > 第三章　刑　　罚 > 第二节　管　　制"
                        "block_types": [block.get("block_type", "text")],
                        "is_ocr": block.get("is_ocr", False),
                        "source_file": block.get("source_file", ""),
                        "strategy": "fixed",
                        "parent_block_idx": block.get("idx", -1),
                    }
                })

    elif strategy == "semantic":
        for chunk in chunk_semantic(blocks):
            raw_chunks.append(chunk)

    elif strategy == "hierarchical":
        for chunk in chunk_hierarchical(blocks):
            raw_chunks.append(chunk)

    else:
        raise ValueError(f"未知策略: {strategy}")

    # 补充公共元信息
    result = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_id = build_chunk_id(raw_name, idx)
        chunk["chunk_id"]              = chunk_id
        chunk["metadata"]["raw_name"] = raw_name
        chunk["metadata"]["strategy"]   = strategy
        chunk["metadata"]["source_file"] = parsed_path.name
        result.append(chunk)

    # 保存
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

    # 合并所有公司的 chunk 到一个文件，方便统一建索引
    combined_path = CHUNKS_DIR / f"all_{STRATEGY}.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    # 简单统计
    avg_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 chunk 长度: {avg_len:.0f} 字符")

    table_count = sum(1 for c in all_chunks if "table" in c["metadata"].get("block_types", []))
    ocr_count   = sum(1 for c in all_chunks if c["metadata"].get("is_ocr"))
    logger.info(f"其中表格块: {table_count}  OCR块: {ocr_count}")


if __name__ == "__main__":
    main()
