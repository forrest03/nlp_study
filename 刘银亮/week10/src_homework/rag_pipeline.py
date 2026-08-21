"""
医疗问答 RAG 流水线

流程：
  (可选) 查询改写
        ↓
  向量检索（阿里百炼 text-embedding-v4 + FAISS）
        +
  BM25 关键词检索（jieba + rank_bm25）
        ↓
  RRF 融合排名
        ↓
  相关性阈值过滤（过低则拒绝回答）
        ↓
  LLM 生成（阿里百炼 qwen-mt-flash）+ 引用标注

使用方式：
  python rag_pipeline.py                            # 交互式
  python rag_pipeline.py --query "小儿肥胖该如何治疗"
  python rag_pipeline.py --query "..." --department "营养保健科"
  python rag_pipeline.py --query "..." --query-rewrite   # 开启查询改写
  python rag_pipeline.py --query "..." --no-bm25         # 关闭 BM25

依赖：
  pip install faiss-cpu rank_bm25 jieba openai numpy
  export ALIYUN_API_KEY="sk-xxx"
  export ALIYUN_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "medical_data" / "vectorstore"
INDEX_PATH      = VECTORSTORE_DIR / "faiss_index.bin"
META_PATH       = VECTORSTORE_DIR / "faiss_meta.json"

ALIYUN_API_URL  = os.getenv("ALIYUN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_MODEL     = "text-embedding-v4"
EMBED_DIM       = 1024
LLM_MODEL       = "qwen-mt-flash"

TOP_K_RETRIEVE  = 10
TOP_K_FINAL     = 4
SCORE_THRESHOLD = 0.25

SYSTEM_PROMPT = """你是一个专业的医疗健康助手，专门回答关于常见疾病、症状和治疗的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体信息时，在句末标注来源编号，如：建议控制饮食[1]
4. 回答简洁明了，重点突出，避免无关废话
5. 仅供参考，不构成医疗建议"""


def get_client() -> OpenAI:
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 ALIYUN_API_KEY")
    return OpenAI(api_key=api_key, base_url=ALIYUN_API_URL)


class VectorStore:
    def __init__(self, client: OpenAI):
        import faiss
        self.client    = client
        self.index     = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)
        logger.info(f"FAISS 索引加载完成，共 {self.index.ntotal} 条向量")

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM
        )
        vec = np.array([resp.data[0].embedding], dtype="float32")
        vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVE,
        filter_meta: Optional[dict] = None,
    ) -> list[dict]:
        query_vec = self._embed_query(query)
        scores, indices = self.index.search(query_vec, top_k * 4)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.meta_list):
                continue
            item = dict(self.meta_list[idx])
            item["vec_score"] = float(score)

            if filter_meta:
                if not all(str(item.get(k, "")) == str(v) for k, v in filter_meta.items()):
                    continue

            results.append(item)
            if len(results) >= top_k:
                break
        return results


class BM25Store:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        import jieba

        with open(META_PATH, encoding="utf-8") as f:
            self.meta_list = json.load(f)

        logger.info("构建 BM25 索引（分词中，请稍候）...")
        tokenized = [list(jieba.cut(item["content"])) for item in self.meta_list]
        self.bm25  = BM25Okapi(tokenized)
        self.jieba = jieba
        logger.info("BM25 索引完成")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
        tokens = list(self.jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] < 1e-9:
                continue
            item = dict(self.meta_list[idx])
            item["bm25_score"] = float(scores[idx])
            results.append(item)
        return results


def reciprocal_rank_fusion(
    vec_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, item in enumerate(vec_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    for rank, item in enumerate(bm25_results, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    results = []
    for cid in sorted_cids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = rrf_scores[cid]
        results.append(item)
    return results


def rewrite_query(query: str, client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是检索查询优化专家。将用户的医疗问题改写为更适合从医疗问答知识库中检索信息的精确查询语句。"
                    "保留关键实体（症状、疾病、治疗方法），扩展相关关键词，不要超过50字。"
                    "直接输出改写后的查询语句，不要解释。"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    rewritten = resp.choices[0].message.content.strip()
    logger.info(f"查询改写: {query!r} → {rewritten!r}")
    return rewritten


def build_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    parts     = []
    citations = []

    for i, item in enumerate(retrieved, 1):
        department = item.get("department", "")
        title      = item.get("title", "")

        label = f"[{i}]"
        if department:
            label += f" {department}"
        if title:
            label += f" · {title}"

        content = item.get("answer") or item.get("content", "")
        parts.append(f"{label}\n{content}")
        citations.append({"index": i, "source": label, "chunk_id": item.get("chunk_id", "")})

    return "\n\n---\n\n".join(parts), citations


def call_llm(query: str, context: str, client: OpenAI) -> str:
    user_msg = (
        f"{SYSTEM_PROMPT}\n\n"
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用信息处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


class RAGPipeline:
    def __init__(
        self,
        use_bm25:        bool = True,
        use_query_rewrite: bool = False,
    ):
        self.client        = get_client()
        self.vec_store     = VectorStore(self.client)
        self.use_bm25      = use_bm25
        self.use_qr        = use_query_rewrite
        self.bm25_store    = BM25Store() if use_bm25 else None

    def query(
        self,
        question: str,
        filter_meta: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        retrieval_query = rewrite_query(question, self.client) if self.use_qr else question

        vec_results = self.vec_store.search(retrieval_query, TOP_K_RETRIEVE, filter_meta)
        if verbose:
            logger.info(f"向量召回: {len(vec_results)} 条，最高分={vec_results[0]['vec_score']:.3f}" if vec_results else "向量召回: 0 条")

        if self.use_bm25 and self.bm25_store:
            bm25_results = self.bm25_store.search(retrieval_query, TOP_K_RETRIEVE)
            candidates   = reciprocal_rank_fusion(vec_results, bm25_results)
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results

        final = candidates[:TOP_K_FINAL]

        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")

        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [], "retrieved": [],
            }

        top_score = final[0].get("vec_score", 1.0)
        if top_score < SCORE_THRESHOLD and filter_meta is None:
            return {
                "answer": "根据医疗知识库未能找到与该问题相关的内容。",
                "citations": [], "retrieved": final,
            }

        context, citations = build_context(final)
        answer = call_llm(question, context, self.client)

        return {"answer": answer, "citations": citations, "retrieved": final}


def main():
    parser = argparse.ArgumentParser(description="医疗问答 RAG（原生版）")
    parser.add_argument("--query",         type=str,  default=None)
    parser.add_argument("--department",    type=str,  default=None, help="科室，如 营养保健科")
    parser.add_argument("--query-rewrite", action="store_true", help="开启查询改写")
    parser.add_argument("--no-bm25",       action="store_true", help="关闭 BM25")
    args = parser.parse_args()

    pipeline = RAGPipeline(
        use_bm25         = not args.no_bm25,
        use_query_rewrite= args.query_rewrite,
    )

    filter_meta = {}
    if args.department: filter_meta["department"] = args.department
    if not filter_meta: filter_meta = None

    def print_result(q: str, result: dict):
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print(f"{'='*60}")
        print(f"\n{result['answer']}")
        if result["citations"]:
            print("\n── 来源 ──")
            for c in result["citations"]:
                print(f"  {c['source']}")

    if args.query:
        result = pipeline.query(args.query, filter_meta=filter_meta, verbose=True)
        print_result(args.query, result)
    else:
        print("医疗问答 RAG 系统（原生版）")
        print(f"模型：{LLM_MODEL}  |  向量库：{INDEX_PATH}")
        print("输入 'exit' 退出，'mode' 查看当前配置\n")
        while True:
            try:
                q = input("问题：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() == "exit":
                break
            if q.lower() == "mode":
                print(f"BM25={'on' if pipeline.use_bm25 else 'off'}  "
                      f"QueryRewrite={'on' if pipeline.use_qr else 'off'}")
                continue
            result = pipeline.query(q, filter_meta=filter_meta, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()