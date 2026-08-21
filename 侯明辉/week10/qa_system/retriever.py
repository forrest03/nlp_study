# week10检索增强生成/qa_system/retriever.py
"""检索：RRF 融合 + 阈值（纯逻辑）与 向量/BM25/retrieve 编排。"""
import json

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

import config


def reciprocal_rank_fusion(ranked_lists, k=None):
    """输入多个已排序的 chunk_id 列表，返回 [(chunk_id, rrf_score)] 按分降序。"""
    k = k or config.RRF_K
    scores = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def passes_threshold(top_vec_score, threshold=None) -> bool:
    """用原始向量相似度（余弦，0~1）判断是否达到回答阈值。"""
    threshold = config.SCORE_THRESHOLD if threshold is None else threshold
    return top_vec_score >= threshold


class Retriever:
    def __init__(self, index_dir=None, embed_fn=None):
        index_dir = index_dir or config.INDEX_DIR
        self.embeddings = np.load(f"{index_dir}/embeddings.npy")
        with open(f"{index_dir}/chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.id2chunk = {c["chunk_id"]: c for c in self.chunks}
        self.ids = [c["chunk_id"] for c in self.chunks]
        # 注入式 embed（测试传假函数；生产用 embedder）
        if embed_fn is None:
            import embedder
            embed_fn = embedder.embed_texts
        self._embed = embed_fn
        # BM25 索引
        tokenized = [list(jieba.cut(c["content"])) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def _vector_scores(self, query):
        qvec = np.asarray(self._embed([query]), dtype=np.float32)[0]
        return self.embeddings @ qvec           # 归一化后内积 = 余弦

    def retrieve(self, query, k_recall=None, k_final=None):
        k_recall = k_recall or config.TOP_K_RECALL
        k_final = k_final or config.TOP_K_FINAL

        vec_scores = self._vector_scores(query)
        vec_order = np.argsort(vec_scores)[::-1][:k_recall]
        vec_ids = [self.ids[i] for i in vec_order]

        bm25_scores = self._bm25.get_scores(list(jieba.cut(query)))
        bm25_order = np.argsort(bm25_scores)[::-1][:k_recall]
        bm25_ids = [self.ids[i] for i in bm25_order]

        fused = reciprocal_rank_fusion([vec_ids, bm25_ids])[:k_final]

        id2vec = {self.ids[i]: float(vec_scores[i]) for i in range(len(self.ids))}
        results = []
        for cid, rrf in fused:
            chunk = dict(self.id2chunk[cid])
            chunk["vec_score"] = id2vec.get(cid, 0.0)
            chunk["rrf_score"] = rrf
            results.append(chunk)
        top_vec = results[0]["vec_score"] if results else 0.0
        return results, top_vec

