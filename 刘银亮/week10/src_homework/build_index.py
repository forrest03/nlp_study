"""
医疗问答数据向量索引构建脚本

Embedding 方案：阿里百炼 text-embedding-v4
  - 模型：text-embedding-v4
  - 维度：1024
  - 平台：阿里百炼

向量库：FAISS（IndexFlatIP，内积 = 归一化后的余弦相似度）

依赖：
  pip install faiss-cpu openai numpy
  export ALIYUN_API_KEY="sk-xxx"
  export ALIYUN_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "medical_data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "medical_data" / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_FILE     = CHUNKS_DIR / "child-5_chunks.json"

EMBED_MODEL     = "text-embedding-v4"
EMBED_DIM       = 1024
BATCH_SIZE      = 10


def get_client() -> OpenAI:
    api_key = os.getenv("ALIYUN_API_KEY")
    api_url = os.getenv("ALIYUN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 ALIYUN_API_KEY\n"
            "  Linux/Mac: export ALIYUN_API_KEY=sk-xxx\n"
            "  Windows: set ALIYUN_API_KEY=sk-xxx"
        )
    return OpenAI(
        api_key=api_key,
        base_url=api_url
    )


def embed_texts(client: OpenAI, texts: list[str], show_progress: bool = True) -> np.ndarray:
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(f"  开始计算 embedding，共 {len(texts)} 条文本，分 {total_batches} 批处理")

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and batch_idx % 100 == 0:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批 ({(batch_idx/total_batches)*100:.1f}%)")

        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                vecs = [e.embedding for e in resp.data]
                all_embeddings.extend(vecs)
                break
            except Exception as e:
                if attempt == 2:
                    logger.error(f"  Embedding 第{batch_idx}批第{attempt+1}次重试仍失败，错误: {e}")
                    raise
                logger.warning(f"  Embedding 第{batch_idx}批第{attempt+1}次失败，重试: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")
    logger.info(f"  Embedding 计算完成，形状: {embeddings.shape}，数据类型: {embeddings.dtype}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings = embeddings / norms
    logger.info(f"  L2 归一化完成，范数范围: [{norms.min():.6f}, {norms.max():.6f}]")

    return embeddings


def build_faiss_index(chunks: list[dict], client: OpenAI):
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    logger.info(f"  第一条 chunk_id: {chunks[0]['chunk_id']}")
    logger.info(f"  第一条 content 长度: {len(chunks[0]['content'])} 字符")
    logger.info(f"  第一条 answer 长度: {len(chunks[0]['answer'])} 字符")

    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(client, texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id":    c["chunk_id"],
            "content":     c["content"],
            "answer":      c["answer"],
            "department":  c["metadata"].get("department", ""),
            "title":       c["metadata"].get("title", ""),
            "source_file": c["metadata"].get("source_file", ""),
            "row_index":   c["metadata"].get("row_index", -1),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


def main():
    logger.info(f"开始构建向量索引")
    logger.info(f"  配置:")
    logger.info(f"    CHUNKS_FILE: {CHUNKS_FILE}")
    logger.info(f"    EMBED_MODEL: {EMBED_MODEL}")
    logger.info(f"    EMBED_DIM:   {EMBED_DIM}")
    logger.info(f"    BATCH_SIZE:  {BATCH_SIZE}")
    logger.info(f"    VECTORSTORE_DIR: {VECTORSTORE_DIR}")

    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk.py")
        return

    logger.info(f"\n加载 chunks 文件...")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"  加载完成，共 {len(chunks)} 个 chunks")

    if chunks:
        first_chunk = chunks[0]
        logger.info(f"  第一条 chunk 结构: {list(first_chunk.keys())}")
        if "metadata" in first_chunk:
            logger.info(f"  第一条 metadata 结构: {list(first_chunk['metadata'].keys())}")

    logger.info(f"\n初始化阿里百炼 OpenAI Client...")
    client = get_client()
    logger.info(f"  Client 初始化完成，base_url: {client.base_url}")

    logger.info(f"\n开始构建 FAISS 索引...")
    build_faiss_index(chunks, client)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()