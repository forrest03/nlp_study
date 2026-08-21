"""
长期记忆压缩模块

定期分析 memory/long_term/memories_raw.md：
  1. 调用 LLM 提取关键事实并压缩
  2. 人类可读结果写入 memory/compressed/memories.md
  3. 检索元数据写入 databases/memory_meta.json
  4. 重建 databases/memory_faiss_index.bin
  5. 清空 raw 缓冲区
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, List

from harness.llm_client import DEFAULT_MODEL, get_client
from src.memory_store import get_memory_store
from src.paths import DATABASE_DIR, FAISS_INDEX_FILE, MEMORY_META_FILE

logger = logging.getLogger(__name__)

COMPRESS_SYSTEM = """你是记忆压缩专家。将用户提供的对话/事件记录压缩为简洁、可检索的记忆条目。

要求：
1. 每条记忆独立成段，保留关键实体（人名、日期、偏好、决策、事实）
2. 去除重复和无关寒暄
3. 每条 1-3 句话，中文输出
4. 直接输出 JSON 数组，格式：[{"summary": "记忆摘要内容", "tags": ["标签1"]}]
5. 不要输出 markdown 代码块，只输出 JSON"""


def compress_raw_memories() -> List[Dict]:
    store = get_memory_store()
    raw = store.read_long_term_raw().strip()
    if not raw:
        logger.info("无待压缩的原始长期记忆")
        return []

    client = get_client()
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": COMPRESS_SYSTEM},
            {"role": "user", "content": f"请压缩以下记忆记录：\n\n{raw}"},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        summaries = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM 返回非 JSON，整段作为单条记忆保存")
        summaries = [{"summary": text, "tags": ["auto"]}]

    chunks = []
    ts = int(time.time())
    for item in summaries:
        summary = item.get("summary", "").strip()
        if not summary:
            continue
        chunks.append({
            "chunk_id": str(uuid.uuid4())[:12],
            "content": summary,
            "tags": item.get("tags", []),
            "created_at": ts,
        })

    if chunks:
        store.save_compressed_chunks(chunks)
        store.clear_long_term_raw()
        rebuild_vector_index()
        logger.info(f"已压缩 {len(chunks)} 条长期记忆，写入 databases/ 检索索引")

    return chunks


def rebuild_vector_index():
    """从 databases/memory_meta.json 重建 FAISS 向量索引。"""
    import faiss
    import numpy as np
    from openai import OpenAI

    store = get_memory_store()
    chunks = store.read_compressed_chunks()
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    if not chunks:
        index = faiss.IndexFlatL2(1024)
        faiss.write_index(index, str(FAISS_INDEX_FILE))
        MEMORY_META_FILE.write_text("[]", encoding="utf-8")
        return

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    embed_model = "text-embedding-v3"
    dim = 1024

    vectors = []
    for chunk in chunks:
        resp = client.embeddings.create(
            model=embed_model,
            input=[chunk["content"]],
            dimensions=dim,
        )
        vec = np.array(resp.data[0].embedding, dtype="float32")
        vec = vec / max(np.linalg.norm(vec), 1e-9)
        vectors.append(vec)

    matrix = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatL2(dim)
    index.add(matrix)

    faiss.write_index(index, str(FAISS_INDEX_FILE))
    MEMORY_META_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"向量索引重建完成（databases/），共 {len(chunks)} 条")


def run_compression_if_needed(min_raw_chars: int = 500, session_id: str = "") -> bool:
    store = get_memory_store()
    if not store.compressed_needs_update(min_raw_chars):
        return False

    if session_id:
        turns = store.get_short_term(session_id)
        if len(turns) < 10:
            return False

    compress_raw_memories()
    return True
