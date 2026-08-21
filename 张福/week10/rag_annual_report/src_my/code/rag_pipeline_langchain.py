"""
RAG 问答流水线（LangChain 版）

基于 LangChain 框架实现，提供与原生版一致的功能：
  - 向量检索（FAISS）
  - BM25 关键词检索
  - RRF 融合
  - CrossEncoder Rerank
  - 查询改写
  - LLM 生成

使用方式：
  python src_my/code/rag_pipeline_langchain.py
  python src_my/code/rag_pipeline_langchain.py --query "2025年节能降碳节能量"
  python src_my/code/rag_pipeline_langchain.py --query "..." --years "2024|2025" --query-rewrite

依赖：
  pip install langchain langchain-community langchain-core faiss-cpu rank_bm25 jieba sentence-transformers openai
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR        = Path(__file__).parent.parent
CHUNKS_DIR      = BASE_DIR / "resource" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "resource" / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY        = "semantic"
CHUNKS_FILE     = CHUNKS_DIR / f"all_{STRATEGY}.json"

TOP_K_RETRIEVE  = 10
TOP_K_RERANK    = 4
SCORE_THRESHOLD = 0.25

DASHSCOPE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL     = "text-embedding-v3"
LLM_MODEL       = "qwen-plus"

SYSTEM_PROMPT = """你是一个专业的政策分析助手，专门回答关于节能降碳、碳足迹核算等政策文件的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的数据
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"
3. 引用具体数据时，在句末标注来源编号，如：节能量约5000万吨标准煤[1]
4. 数字要精确，不得四舍五入或模糊表达
5. 回答简洁，重点突出，避免无关废话"""


def get_dashscope_api_key(api_key: str = None) -> str:
    if not api_key:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置 API Key\n"
            "  方式1: 环境变量 export DASHSCOPE_API_KEY=sk-xxx\n"
            "  方式2: 命令行参数 --api-key sk-xxx"
        )
    return api_key


def load_chunks() -> List[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"找不到分块文件: {CHUNKS_FILE}\n请先运行: python src_my/code/chunk_documents.py")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


class DashScopeEmbeddings:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)
        self.model = EMBED_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        batch_size = 10
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch, dimensions=1024)
            vecs = [e.embedding for e in resp.data]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            vecs = (np.array(vecs) / norms).tolist()
            all_embeddings.extend(vecs)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


def build_vectorstore(chunks: List[dict], api_key: str):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    vectorstore_path = VECTORSTORE_DIR / "faiss_langchain"

    if vectorstore_path.exists():
        logger.info("加载已存在的 FAISS 向量库...")
        embeddings = DashScopeEmbeddings(api_key)
        return FAISS.load_local(str(vectorstore_path), embeddings, allow_dangerous_deserialization=True)

    logger.info("构建 FAISS 向量库（首次运行会调用 API 生成 embedding）...")
    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "chunk_id":       chunk["chunk_id"],
                "page_num":       chunk["metadata"].get("page_num", -1),
                "section":        chunk["metadata"].get("section", ""),
                "block_types":    chunk["metadata"].get("block_types", []),
                "is_ocr":         chunk["metadata"].get("is_ocr", False),
                "strategy":       chunk["metadata"].get("strategy", ""),
                "source_file":    chunk["metadata"].get("source_file", ""),
                "parent_content": chunk["metadata"].get("parent_content", ""),
                "parent_id":      chunk["metadata"].get("parent_id", ""),
            }
        )
        docs.append(doc)

    embeddings = DashScopeEmbeddings(api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(vectorstore_path))
    logger.info(f"FAISS 向量库已保存 → {vectorstore_path}")
    return vectorstore


def build_bm25_retriever(chunks: List[dict]):
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"]
        )
        docs.append(doc)

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = TOP_K_RETRIEVE
    return retriever


def reciprocal_rank_fusion(results: List[List], k: int = 60):
    from langchain_core.documents import Document

    rrf_scores = {}
    doc_map = {}

    for retriever_results in results:
        for rank, doc in enumerate(retriever_results, 1):
            doc_id = doc.metadata.get("chunk_id", id(doc))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    fused_docs = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc.metadata["rrf_score"] = rrf_scores[doc_id]
        fused_docs.append(doc)
    return fused_docs


def rerank_docs(query: str, docs: List, top_k: int = TOP_K_RERANK):
    try:
        from sentence_transformers import CrossEncoder

        model_path = Path(__file__).parent.parent.parent / "models" / "BAAI--bge-reranker-base" / "snapshots" / "master"
        model_name = str(model_path) if model_path.exists() else "BAAI/bge-reranker-base"

        reranker = CrossEncoder(model_name)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: -x[1])
        return [doc for doc, score in scored_docs[:top_k]]
    except ImportError:
        logger.warning("sentence-transformers 未安装，跳过 Rerank")
        return docs[:top_k]
    except Exception as e:
        logger.warning(f"Rerank 失败: {e}")
        return docs[:top_k]


def rewrite_query(query: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)
    resp = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是检索查询优化专家。将用户的问题改写为更适合从政策文件中检索信息的精确查询语句。"
                    "保留关键实体（年份、政策名称、指标），扩展相关关键词，不要超过50字。"
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


def build_context(docs: List) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source_file = doc.metadata.get("source_file", "")
        page = doc.metadata.get("page_num", "")
        section = doc.metadata.get("section", "")

        label = f"[{i}] {source_file}"
        if section:
            label += f" · {section[:50]}"
        if page and page != -1:
            label += f" · 第{page}页"

        content = doc.metadata.get("parent_content") or doc.page_content
        parts.append(f"{label}\n{content}")

    return "\n\n---\n\n".join(parts)


def call_llm(query: str, context: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)

    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用数据处标注来源编号（如[1]）。"
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def filter_docs_by_years(docs: List, years: List[str]) -> List:
    filtered = []
    for doc in docs:
        source_file = doc.metadata.get("source_file", "")
        for year in years:
            if str(year) in source_file:
                filtered.append(doc)
                break
    return filtered


class LangChainRAGPipeline:
    def __init__(
        self,
        api_key: str,
        use_bm25: bool = True,
        use_rerank: bool = True,
        use_query_rewrite: bool = False,
    ):
        self.api_key = api_key
        self.use_bm25 = use_bm25
        self.use_rerank = use_rerank
        self.use_qr = use_query_rewrite

        self.chunks = load_chunks()
        logger.info(f"加载 {len(self.chunks)} 个 chunks")

        self.vectorstore = build_vectorstore(self.chunks, api_key)
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})

        self.bm25_retriever = build_bm25_retriever(self.chunks) if use_bm25 else None

    def query(
        self,
        question: str,
        years: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> dict:
        retrieval_query = rewrite_query(question, self.api_key) if self.use_qr else question

        vec_results = self.vector_retriever.invoke(retrieval_query)
        if verbose:
            logger.info(f"向量召回: {len(vec_results)} 条")

        if self.use_bm25 and self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(retrieval_query)
            candidates = reciprocal_rank_fusion([vec_results, bm25_results])
            if verbose:
                logger.info(f"BM25 召回: {len(bm25_results)} 条，RRF 后: {len(candidates)} 条")
        else:
            candidates = vec_results

        if years:
            candidates = filter_docs_by_years(candidates, years)
            if verbose:
                logger.info(f"年份过滤后: {len(candidates)} 条")

        if self.use_rerank:
            final = rerank_docs(question, candidates, TOP_K_RERANK)
        else:
            final = candidates[:TOP_K_RERANK]

        if verbose:
            logger.info(f"最终使用 {len(final)} 条上下文")

        if not final:
            return {
                "answer": "未找到相关内容，无法回答此问题。",
                "citations": [],
                "retrieved": [],
            }

        context = build_context(final)
        answer = call_llm(question, context, self.api_key)

        citations = []
        for i, doc in enumerate(final, 1):
            source_file = doc.metadata.get("source_file", "")
            page = doc.metadata.get("page_num", "")
            section = doc.metadata.get("section", "")

            label = f"[{i}] {source_file}"
            if section:
                label += f" · {section[:50]}"
            if page and page != -1:
                label += f" · 第{page}页"

            citations.append({"index": i, "source": label, "chunk_id": doc.metadata.get("chunk_id", "")})

        return {"answer": answer, "citations": citations, "retrieved": final}


def main():
    parser = argparse.ArgumentParser(description="RAG 问答系统（LangChain 版）")
    parser.add_argument("--query",         type=str,  default=None)
    parser.add_argument("--years",         type=str,  default=None, help="年份过滤，多个用|分隔")
    parser.add_argument("--api-key",       type=str,  default=None, help="DashScope API Key")
    parser.add_argument("--query-rewrite", action="store_true", help="开启查询改写")
    parser.add_argument("--no-bm25",       action="store_true", help="关闭 BM25")
    parser.add_argument("--no-rerank",     action="store_true", help="关闭 Rerank")
    args = parser.parse_args()

    api_key = get_dashscope_api_key(args.api_key)

    pipeline = LangChainRAGPipeline(
        api_key=api_key,
        use_bm25=not args.no_bm25,
        use_rerank=not args.no_rerank,
        use_query_rewrite=args.query_rewrite,
    )

    years_filter = args.years.split("|") if args.years else None

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
        result = pipeline.query(args.query, years=years_filter, verbose=True)
        print_result(args.query, result)
    else:
        print("RAG 问答系统（LangChain 版）")
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
                      f"Rerank={'on' if pipeline.use_rerank else 'off'}  "
                      f"QueryRewrite={'on' if pipeline.use_qr else 'off'}")
                continue
            result = pipeline.query(q, years=years_filter, verbose=True)
            print_result(q, result)


if __name__ == "__main__":
    main()