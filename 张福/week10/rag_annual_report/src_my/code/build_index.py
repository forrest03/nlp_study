"""
向量索引构建脚本：通过 DashScope API 构建 FAISS 向量索引

内部流程：
  1. 加载 src_my/resource/chunks/all_semantic.json
  2. 按批次（10条/批）调用 DashScope text-embedding-v3
  3. 对所有向量做 L2 归一化（使内积等价于余弦相似度）
  4. 构建 FAISS IndexFlatIP，批量 add 归一化向量
  5. 分别保存索引文件和元数据

配置方式：
  环境变量配置：
    export DASHSCOPE_API_KEY="sk-xxx"

  或在运行时通过参数配置：
    python build_index.py --api-key sk-xxx

依赖包：
  faiss-cpu>=1.7.4          # FAISS 向量库（CPU版）
  openai>=1.30.0            # DashScope 兼容 OpenAI 接口
  numpy>=1.24.0             # 向量计算
  argparse                  # 命令行参数解析（内置）

运行方式：
  conda activate py312
  export DASHSCOPE_API_KEY="sk-xxx"
  python src_my/code/build_index.py

输出文件：
  src_my/resource/vectorstore/faiss_index.bin    # FAISS 索引文件
  src_my/resource/vectorstore/faiss_meta.json    # 元数据文件
"""

import os
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "resource" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "resource" / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY        = "semantic"
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"

EMBED_MODEL     = "text-embedding-v3"
EMBED_DIM       = 1024
BATCH_SIZE      = 10
DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_client(api_key: str = None) -> OpenAI:
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置 API Key\n"
            "  方式1: 环境变量 export DASHSCOPE_API_KEY=sk-xxx\n"
            "  方式2: 命令行参数 --api-key sk-xxx"
        )
    return OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)


def embed_texts(client: OpenAI, texts: list[str], show_progress: bool = True) -> np.ndarray:
    """
    批量计算 embedding，每批最多 10 条（DashScope 限制）。
    返回 shape=(N, EMBED_DIM) 的 float32 数组，已 L2 归一化。
    """
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(f"开始计算 embedding，共 {len(texts)} 条，{total_batches} 批")

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and batch_idx % 10 == 0:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    dimensions=EMBED_DIM,
                )
                vecs = [e.embedding for e in resp.data]
                all_embeddings.extend(vecs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"  第{attempt+1}次失败，重试: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings = embeddings / norms

    return embeddings


def build_faiss_index(chunks: list[dict], client: OpenAI):
    """
    构建 FAISS 向量索引。
    IndexFlatIP = 暴力内积检索，精确但不近似。
    数据量 < 10 万时速度完全够用。
    """
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
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
            "chunk_id":       c["chunk_id"],
            "content":        c["content"],
            "page_num":       c["metadata"].get("page_num", -1),
            "section":        c["metadata"].get("section", ""),
            "block_types":    c["metadata"].get("block_types", []),
            "is_ocr":         c["metadata"].get("is_ocr", False),
            "strategy":       c["metadata"].get("strategy", ""),
            "source_file":    c["metadata"].get("source_file", ""),
            "parent_content": c["metadata"].get("parent_content", ""),
            "parent_id":      c["metadata"].get("parent_id", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


def main():
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--api-key", type=str, help="DashScope API Key")
    args = parser.parse_args()

    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    client = get_client(args.api_key)
    build_faiss_index(chunks, client)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()