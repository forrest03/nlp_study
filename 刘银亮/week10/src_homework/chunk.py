"""
医疗问答数据分块脚本

数据来源：medical_data 目录下的 CSV 文件
- child-5.csv: 全量数据，约 117099 条
- example.csv: 示例数据

CSV 结构：department, title, ask, answer

Chunk 策略：每行问答对直接作为一个独立 chunk
- content: title + ask（用于 embedding 检索）
- answer: 医生回答（供 LLM 生成回答时参考）
- metadata: department, title, source_file, row_index

输出目录：medical_data/chunks
"""

import json
import csv
import logging
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEDICAL_DATA_DIR = Path(__file__).parent.parent / "medical_data"
CHUNKS_DIR = MEDICAL_DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def build_chunk_id(source_name: str, idx: int) -> str:
    name = source_name.replace(".csv", "").replace("-", "")
    return f"{name}_{idx:06d}"


def process_csv(csv_path: Path) -> Iterator[dict]:
    """逐行读取 CSV，生成 chunk"""
    with open(csv_path, "r", encoding="gb18030", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            department = row.get("department", "").strip()
            title = row.get("title", "").strip()
            ask = row.get("ask", "").strip()
            answer = row.get("answer", "").strip()

            content = f"{title}。{ask}" if title else ask

            yield {
                "chunk_id": build_chunk_id(csv_path.stem, idx),
                "content": content,
                "answer": answer,
                "metadata": {
                    "department": department,
                    "title": title,
                    "source_file": csv_path.name,
                    "row_index": idx,
                },
            }


def process_file(csv_path: Path):
    logger.info(f"处理 {csv_path.name}")

    chunks = []
    progress_interval = 10000
    total_count = 0

    for chunk in process_csv(csv_path):
        chunks.append(chunk)
        total_count += 1

        if total_count % progress_interval == 0:
            logger.info(f"  已处理 {total_count} 条记录")

    out_path = CHUNKS_DIR / f"{csv_path.stem}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"  → {len(chunks)} 个 chunk，已保存 {out_path.name}")
    return chunks


def main():
    csv_files = [f for f in MEDICAL_DATA_DIR.glob("*.csv") if f.name != "example.csv"]
    if not csv_files:
        logger.error("没有找到 CSV 文件")
        return

    all_chunks = []
    for path in csv_files:
        chunks = process_file(path)
        all_chunks.extend(chunks)

    combined_path = CHUNKS_DIR / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"\n合并完成：共 {len(all_chunks)} 个 chunk → {combined_path}")

    avg_content_len = sum(len(c["content"]) for c in all_chunks) / max(len(all_chunks), 1)
    avg_answer_len = sum(len(c["answer"]) for c in all_chunks) / max(len(all_chunks), 1)
    logger.info(f"平均 content 长度: {avg_content_len:.0f} 字符")
    logger.info(f"平均 answer 长度: {avg_answer_len:.0f} 字符")

    departments = {c["metadata"]["department"] for c in all_chunks}
    logger.info(f"涉及科室: {len(departments)} 个")


if __name__ == "__main__":
    main()