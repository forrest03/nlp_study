"""
长期记忆检索模块

从 databases/ 加载 BM25 + FAISS 向量索引，RRF 融合排名，取 Top-K 整理为提示词。
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

from harness.llm_client import get_client
from src.paths import FAISS_INDEX_FILE, MEMORY_META_FILE

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
TOP_K = 3
RRF_K = 60


def reciprocal_rank_fusion(
    vec_results: List[Dict],
    bm25_results: List[Dict],
    k: int = RRF_K,
) -> List[Dict]:
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


class MemoryRetriever:
    """长期记忆 BM25 + 向量混合检索器（数据源：databases/）。"""

    def __init__(self):
        self._faiss_index = None
        self._meta_list: List[Dict] = []
        self._bm25 = None
        self._jieba = None
        self._client = None

    def _load(self):
        if self._faiss_index is not None:
            return

        import faiss

        self._client = get_client()

        if FAISS_INDEX_FILE.exists() and MEMORY_META_FILE.exists():
            self._faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
            self._meta_list = json.loads(MEMORY_META_FILE.read_text(encoding="utf-8"))
            if self._meta_list:
                self._build_bm25()
        else:
            self._faiss_index = faiss.IndexFlatL2(EMBED_DIM)
            self._meta_list = []

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        import jieba

        self._jieba = jieba
        tokenized = [list(jieba.cut(item["content"])) for item in self._meta_list]
        self._bm25 = BM25Okapi(tokenized)

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        if self._faiss_index.ntotal == 0:
            return []

        resp = self._client.embeddings.create(
            model=EMBED_MODEL,
            input=[query],
            dimensions=EMBED_DIM,
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        vec = vec / max(np.linalg.norm(vec), 1e-9)

        scores, indices = self._faiss_index.search(vec, min(top_k * 3, self._faiss_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta_list):
                continue
            item = dict(self._meta_list[idx])
            item["vec_score"] = float(score)
            results.append(item)
        return results[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        if not self._bm25 or not self._meta_list:
            return []

        tokens = list(self._jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:
                continue
            item = dict(self._meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        self._load()

        if not self._meta_list:
            return []

        vec_results = self._vector_search(query, top_k)
        bm25_results = self._bm25_search(query, top_k)

        if vec_results and bm25_results:
            fused = reciprocal_rank_fusion(vec_results, bm25_results)
        elif vec_results:
            fused = vec_results
        else:
            fused = bm25_results

        return fused[:top_k]

    def build_memory_prompt(self, query: str, top_k: int = TOP_K) -> str:
        hits = self.search(query, top_k)
        if not hits:
            return ""

        lines = ["【相关长期记忆】（BM25 + 向量检索融合，Top-3）"]
        for i, item in enumerate(hits, 1):
            tags = ", ".join(item.get("tags", []))
            score = item.get("rrf_score", item.get("vec_score", 0))
            lines.append(f"{i}. [{tags}] {item['content']}  (融合得分参考: {score:.4f})")
        return "\n".join(lines)


_retriever: Optional[MemoryRetriever] = None


def get_memory_retriever() -> MemoryRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MemoryRetriever()
    return _retriever


def reload_retriever():
    global _retriever
    _retriever = MemoryRetriever()
