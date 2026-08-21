"""
向量索引构建脚本（原生实现）

Embedding 方案：远程 embedding 网关接口
  - 接口契约参照 embedding-server/scripts/test_remote_api.py
  - POST JSON 到 embed_rerank_server.v1.embeddings
  - 请求体核心字段：input

向量库：FAISS（IndexFlatIP，内积 = 归一化后的余弦相似度）

依赖：
  pip install faiss-cpu numpy
  export EMBEDDING_API_BASE_URL="https://open-inner.yzwqa.cn/api"
  export EMBEDDING_API_APP_KEY="your-app-key"
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict
import urllib.parse
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY        = "semantic"          # 与 chunk_documents.py 保持一致
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"

EMBED_DIM       = 1024
BATCH_SIZE      = 24
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_VERSION = "1.0"
METHOD_EMBEDDINGS = "embed_rerank_server.v1.embeddings"


# ── 远程 Embedding 客户端 ─────────────────────────────────────────────────────

def get_embedding_settings() -> Dict[str, str]:
    """读取远程 embedding 网关配置。"""
    base_url = "https://open-inner.yzwqa.cn/api"
    app_key = "WL2y70uf"
    version = "1.0"

    return {
        "base_url": base_url,
        "app_key": app_key,
        "version": version or DEFAULT_VERSION,
    }


def build_url(settings: Dict[str, str], method_name: str) -> str:
    """拼接带固定网关参数的完整 URL。"""
    query = urllib.parse.urlencode(
        {
            "appkey": settings["app_key"],
            "version": settings["version"],
            "method": method_name,
        }
    )
    return f"{settings['base_url']}?{query}"


def post_json(settings: Dict[str, str], method_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """调用远程 JSON 接口并返回解析后的响应体。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=build_url(settings, method_name),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        content = response.read().decode("utf-8")
    return json.loads(content)


# ── Embedding ─────────────────────────────────────────────────────────────────

def parse_embedding_response(response: Dict[str, Any]) -> list[list[float]]:
    """从远程接口响应中提取 embedding 列表。"""
    data = response.get("data")
    if not isinstance(data, list):
        raise ValueError(f"embedding 响应缺少 data 列表: {response}")

    vectors = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list):
            raise ValueError(f"embedding 响应项缺少 embedding 字段: {item}")
        vectors.append(embedding)
    return vectors


def embed_texts(settings: Dict[str, str], texts: list[str], show_progress: bool = True) -> np.ndarray:
    """
    批量计算 embedding。
    返回 shape=(N, EMBED_DIM) 的 float32 数组，已 L2 归一化。
    """
    all_embeddings = []
    total_batches  = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1

        if show_progress and batch_idx % 100 == 0:
            logger.info(f"  Embedding 进度: {batch_idx}/{total_batches} 批")

        for attempt in range(3):
            try:
                response = post_json(settings, METHOD_EMBEDDINGS, {"input": batch})
                vecs = parse_embedding_response(response)
                if len(vecs) != len(batch):
                    raise ValueError(
                        f"embedding 数量不匹配，期望 {len(batch)} 条，实际 {len(vecs)} 条"
                    )
                all_embeddings.extend(vecs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f"  第{attempt+1}次失败，重试: {e}")
                time.sleep(2 ** attempt)

    embeddings = np.array(all_embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
        raise ValueError(f"embedding 维度异常，期望 (*, {EMBED_DIM})，实际 {embeddings.shape}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embeddings = embeddings / norms

    return embeddings


# ── FAISS 索引构建 ─────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict], settings: Dict[str, str]):
    """
    构建 FAISS 向量索引。

    FAISS 说明：
      IndexFlatIP = 暴力内积检索，精确但不近似。
      数据量 < 10 万时速度完全够用，是教学的首选。
      数据量更大时可换 IndexIVFFlat（需要 train）或 IndexHNSW。
    """
    import faiss

    logger.info(f"开始计算 {len(chunks)} 条 chunk 的 embedding...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(settings, texts)

    logger.info(f"构建 FAISS 索引，维度={EMBED_DIM}...")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    logger.info(f"索引构建完成，共 {index.ntotal} 条向量")

    # 持久化：索引文件 + 元数据（分开存，避免把大向量序列化进 JSON）
    index_path = VECTORSTORE_DIR / "faiss_index.bin"
    meta_path  = VECTORSTORE_DIR / "faiss_meta.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS 索引已保存 → {index_path}  ({index_path.stat().st_size//1024} KB)")

    meta_list = [
        {
            "chunk_id":   c["chunk_id"],
            "content":    c["content"],
            "stock_code": c["metadata"].get("stock_code", ""),
            "year":       c["metadata"].get("year", ""),
            "page_num":   c["metadata"].get("page_num", -1),
            "section":    c["metadata"].get("section", ""),
            "block_types":c["metadata"].get("block_types", []),
            "is_ocr":     c["metadata"].get("is_ocr", False),
            "strategy":   c["metadata"].get("strategy", ""),
            "source_file":c["metadata"].get("source_file", ""),
            # 层级分块时保留父块内容供 LLM 读取
            "parent_content": c["metadata"].get("parent_content", ""),
            "parent_id":      c["metadata"].get("parent_id", ""),
        }
        for c in chunks
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    logger.info(f"元数据已保存 → {meta_path}")

    return index, meta_list


# ── ChromaDB 索引构建（可选对比） ──────────────────────────────────────────────

def build_chroma_index(chunks: list[dict], settings: Dict[str, str]):
    """
    ChromaDB 版本（可选）。
    优势：内置元数据过滤，可直接 where={"stock_code": "600519"} 过滤。
    劣势：写入较慢，不适合大批量。
    """
    try:
        import chromadb
    except ImportError:
        logger.error("请先安装 chromadb: pip install chromadb")
        return

    chroma_dir = VECTORSTORE_DIR / "chroma"
    client_db  = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client_db.get_or_create_collection(
        name="annual_reports",
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"向 ChromaDB 写入 {len(chunks)} 条 chunk...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(settings, texts)

    for i in range(0, len(chunks), 100):
        batch   = chunks[i:i+100]
        ids     = [c["chunk_id"] for c in batch]
        docs    = [c["content"] for c in batch]
        embs    = embeddings[i:i+100].tolist()
        metas   = []
        for c in batch:
            m = dict(c["metadata"])
            # ChromaDB 只支持 str/int/float/bool 类型的 metadata value
            m["block_types"] = ",".join(m.get("block_types") or [])
            m.pop("parent_content", None)   # 太长
            metas.append(m)
        collection.add(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
        logger.info(f"  已写入 {min(i+100, len(chunks))}/{len(chunks)}")

    logger.info(f"ChromaDB 写入完成，共 {collection.count()} 条")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    if not CHUNKS_FILE.exists():
        logger.error(f"找不到 {CHUNKS_FILE}，请先运行 chunk_documents.py")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"加载 {len(chunks)} 个 chunks（策略={STRATEGY}）")

    settings = get_embedding_settings()

    build_faiss_index(chunks, settings)

    # build_chroma_index(chunks, settings)

    logger.info("\n索引构建完成！")
    logger.info(f"  FAISS 索引: {VECTORSTORE_DIR / 'faiss_index.bin'}")
    logger.info(f"  元数据:     {VECTORSTORE_DIR / 'faiss_meta.json'}")


if __name__ == "__main__":
    main()
