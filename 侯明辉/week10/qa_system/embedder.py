# week10检索增强生成/qa_system/embedder.py
"""硅基流动（OpenAI 兼容）embedding 封装：分批 + L2 归一化。"""
import numpy as np
from openai import OpenAI

import config


def get_client() -> OpenAI:
    return OpenAI(api_key=config.get_api_key(), base_url=config.API_BASE_URL)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)      # 防除零（week10 技术点）
    return mat / norms


def embed_texts(texts, client=None) -> np.ndarray:
    """返回形状 (len(texts), EMBED_DIM) 的 float32、已 L2 归一化数组。"""
    client = client or get_client()
    vecs = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[i:i + config.EMBED_BATCH_SIZE]
        resp = client.embeddings.create(
            model=config.EMBED_MODEL,
            input=batch,
        )
        vecs.extend(e.embedding for e in resp.data)
    mat = np.asarray(vecs, dtype=np.float32)
    return _l2_normalize(mat)