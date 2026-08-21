"""
会话对话历史向量存储模块

功能：
  1. 将问答对按会话(session_id)进行向量保存
  2. 根据会话ID和当前问题检索相关历史问答
  3. 提供会话管理功能（列出会话、获取会话详情）

使用方式：
  from conversation_store import ConversationStore
  store = ConversationStore()
  store.add(session_id="sess001", question="茅台毛利率", answer="茅台2023年毛利率91.4%")
  history = store.search(session_id="sess001", query="茅台财务数据", top_k=3)
"""

import os
import json
import uuid
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import faiss
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── 配置 ────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CONV_VECTORSTORE_DIR = BASE_DIR / "conv_vectorstore"
CONV_VECTORSTORE_DIR.mkdir(exist_ok=True)

# 向量存储文件名
FAISS_INDEX_FILE = CONV_VECTORSTORE_DIR / "conv_faiss_index.bin"
META_FILE = CONV_VECTORSTORE_DIR / "conv_faiss_meta.json"

# Embedding 客户端（与 tools.py 保持一致）
_embed_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
EMBED_MODEL = "text-embedding-v3"


class ConversationStore:
    """会话历史向量存储"""

    def __init__(self):
        self._faiss_index = None
        self._faiss_meta = None
        self._dimension = 1024

    def _load(self):
        """加载或初始化FAISS索引和元数据"""
        if self._faiss_index is not None:
            return

        if FAISS_INDEX_FILE.exists() and META_FILE.exists():
            logger.info("加载会话FAISS索引...")
            self._faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
            with open(META_FILE, encoding="utf-8") as f:
                self._faiss_meta = json.load(f)
            logger.info(f"会话索引就绪，共 {self._faiss_index.ntotal} 条记录")
        else:
            logger.info("初始化新的会话FAISS索引...")
            self._faiss_index = faiss.IndexFlatL2(self._dimension)
            self._faiss_meta = []

    def _embed(self, text: str) -> np.ndarray:
        """对文本进行向量编码"""
        resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=[text])
        vec = np.array(resp.data[0].embedding, dtype="float32")
        vec = vec / np.linalg.norm(vec)
        return vec.reshape(1, -1)

    def _save(self):
        """保存FAISS索引和元数据到磁盘"""
        faiss.write_index(self._faiss_index, str(FAISS_INDEX_FILE))
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(self._faiss_meta, f, ensure_ascii=False, indent=2)

    def add(self, session_id: str, question: str, answer: str):
        """
        添加一条问答记录到向量存储

        Args:
            session_id: 会话ID
            question: 用户问题
            answer: 回答内容
        """
        self._load()

        vec = self._embed(question)

        self._faiss_index.add(vec)

        meta = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "timestamp": int(time.time()),
        }
        self._faiss_meta.append(meta)

        self._save()
        logger.info(f"已保存问答记录，session_id={session_id}")

    def search(self, session_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        在指定会话中检索相关历史问答

        Args:
            session_id: 会话ID
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            匹配的历史问答列表，按相似度降序排列
        """
        self._load()

        if self._faiss_index.ntotal == 0:
            return []

        query_vec = self._embed(query)

        scores, indices = self._faiss_index.search(query_vec, top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._faiss_meta):
                continue
            meta = self._faiss_meta[idx]
            if meta["session_id"] == session_id:
                results.append({
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "score": float(score),
                })

        results.sort(key=lambda x: x["score"])
        return results[:top_k]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        列出所有会话

        Returns:
            会话列表，每个会话包含session_id、问答数量和创建时间，按创建时间倒序排列
        """
        self._load()

        session_stats = {}
        for meta in self._faiss_meta:
            sess_id = meta["session_id"]
            if sess_id not in session_stats:
                session_stats[sess_id] = {
                    "session_id": sess_id,
                    "count": 0,
                    "first_question": meta["question"],
                    "created_at": meta["timestamp"],
                }
            session_stats[sess_id]["count"] += 1

        return sorted(session_stats.values(), key=lambda x: x["created_at"], reverse=True)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取指定会话的所有历史记录

        Args:
            session_id: 会话ID

        Returns:
            该会话的所有问答记录列表，按时间顺序排列
        """
        self._load()

        history = []
        for meta in self._faiss_meta:
            if meta["session_id"] == session_id:
                history.append({
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "timestamp": meta["timestamp"],
                })

        history.sort(key=lambda x: x["timestamp"])
        return history

    def create_session(self) -> str:
        """
        创建一个新会话，返回会话ID

        Returns:
            新的会话ID
        """
        self._load()

        session_id = str(uuid.uuid4())[:8]

        placeholder_vec = np.zeros((1, self._dimension), dtype="float32")
        self._faiss_index.add(placeholder_vec)

        meta = {
            "session_id": session_id,
            "question": "新会话",
            "answer": "等待提问...",
            "timestamp": int(time.time()),
        }
        self._faiss_meta.append(meta)

        self._save()
        logger.info(f"创建新会话: {session_id}")
        return session_id


_conv_store = ConversationStore()


def get_conv_store() -> ConversationStore:
    """获取全局会话存储实例"""
    return _conv_store