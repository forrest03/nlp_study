#!/usr/bin/env python3
"""
RAG 法规检索工具 — 两种模式:
  交互式: python3 search_cli.py              启动后设定参数，循环输入查询
  单次查询: python3 search_cli.py "查询词"    一行出结果 (可加 -k 10 -d vector_db/)

快捷入口: ./search  →  ./search "查询词"
"""
import sys, os, json, glob, readline
import numpy as np

DEFAULT_DB = "知识库原始文件/vector_db/"
DEFAULT_K = 5


def find_db_dir():
    """智能定位 vector_db"""
    for root, dirs, files in os.walk("."):
        if "faiss.index" in files:
            return root
    for root, dirs, files in os.walk(os.path.dirname(__file__) or "."):
        if "faiss.index" in files:
            return root
    return None


def load(db_dir: str):
    import faiss
    from sentence_transformers import SentenceTransformer

    model_name = open(os.path.join(db_dir, "model_name.txt")).read().strip()
    model = SentenceTransformer(model_name)
    index = faiss.read_index(os.path.join(db_dir, "faiss.index"))
    meta = json.load(open(os.path.join(db_dir, "meta.json")))

    info = {
        "model": model_name,
        "dim": model.get_embedding_dimension(),
        "vectors": index.ntotal,
        "chunks": len(meta),
    }
    return model, index, meta, info


def search(model, index, meta, query: str, top_k: int = 5):
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_vec, top_k)
    return [{**meta[idx], "score": float(s)} for s, idx in zip(D[0], I[0])]


def fmt_result(r, i: int):
    return (
        f"\n{'─'*60}"
        f"\n[{i}] score={r['score']:.4f} | {r['doc_name']} | {r['chapter']}"
        f"\n    chunk_id: {r['chunk_id']} ({r['chunk_type']})"
        f"\n    article_range: {r['article_range']}"
        f"\n{'-'*60}"
        f"\n{r['text']}"
    )


def interactive(model, index, meta, top_k):
    """交互循环"""
    print("输入中文查询 (q=退出, k=N=调top_k, stat=统计, doc=列文档)\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break
        if q == "?":
            print("查询示例: 金融租赁公司注册资本要求")
            print("命令: q=退出  k=N=条数  stat=统计  doc=列文档")
            continue
        if q.startswith("k="):
            try:
                top_k = int(q.split("=", 1)[1])
                print(f"  top_k → {top_k}")
            except ValueError:
                pass
            continue
        if q == "stat":
            from collections import Counter
            docs = Counter(r["doc_name"] for r in meta)
            types = Counter(r["chunk_type"] for r in meta)
            print(f"  总 chunks: {len(meta)}")
            for d, n in docs.most_common():
                print(f"    {d}: {n}")
            print(f"  类型: {dict(types)}")
            continue
        if q == "doc":
            for d in sorted(set(r["doc_name"] for r in meta)):
                n = sum(1 for r in meta if r["doc_name"] == d)
                print(f"  {d} ({n} chunks)")
            continue

        results = search(model, index, meta, q, top_k)
        print(f"\n查询: {q}")
        for i, r in enumerate(results, 1):
            print(fmt_result(r, i))


def once(model, index, meta, query, top_k):
    """单次查询模式"""
    results = search(model, index, meta, query, top_k)
    print(f"查询: {query}  (top_k={top_k})\n")
    for i, r in enumerate(results, 1):
        print(fmt_result(r, i))
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="RAG 法规检索")
    ap.add_argument("query", nargs="?", help="查询词（不传则进入交互模式）")
    ap.add_argument("-k", "--top-k", type=int, default=DEFAULT_K)
    ap.add_argument("-d", "--db", default=None, help="vector_db 路径")
    args = ap.parse_args()

    db_dir = args.db or find_db_dir() or DEFAULT_DB
    if not os.path.exists(os.path.join(db_dir, "faiss.index")):
        print(f"错误: {db_dir} 下没有 faiss.index")
        sys.exit(1)

    model, index, meta, info = load(db_dir)
    print(f"模型={info['model']} 维度={info['dim']} 向量={info['vectors']} chunks={info['chunks']}\n")

    if args.query:
        once(model, index, meta, args.query, args.top_k)
    else:
        interactive(model, index, meta, args.top_k)


if __name__ == "__main__":
    main()