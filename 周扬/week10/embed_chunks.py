#!/usr/bin/env python3
"""
chunks → embeddings → FAISS 向量库
用法: python3 embed_chunks.py 知识库原始文件/  [--model BAAI/bge-small-zh-v1.5]
"""
import sys, os, json, argparse, glob
import numpy as np

def load_all_chunks(chunks_dir: str) -> list:
    """加载知识库目录下所有 *_chunks.jsonl"""
    all_chunks = []
    for f in sorted(glob.glob(os.path.join(chunks_dir, "*_chunks.jsonl"))):
        doc_name = os.path.basename(f).replace("_chunks.jsonl", "")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                c["doc_name"] = doc_name
                # 仅可检索的 chunk 参与 embedding
                if c["chunk_type"] != "chapter_summary":
                    all_chunks.append(c)
    return all_chunks


def build_index(chunks: list, model_name: str, out_dir: str):
    """embed + FAISS 索引"""
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"dim:   {dim}")

    texts = [c["text"] for c in chunks]
    print(f"chunks: {len(texts)}")

    print("embedding...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS 内积索引（归一化后等价于余弦相似度）
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"index: {index.ntotal} vectors")

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    print(f"saved: {out_dir}/faiss.index")

    # 保存 chunk 元数据（检索时回查原文）
    meta = [{
        "id": i,
        "chunk_id": c["chunk_id"],
        "chunk_type": c["chunk_type"],
        "doc_name": c.get("doc_name", ""),
        "chapter": c.get("chapter", ""),
        "article_range": c.get("article_range", ""),
        "text": c["text"],
    } for i, c in enumerate(chunks)]
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_dir}/meta.json")

    # 保存 model_name 以便检索时复用
    with open(os.path.join(out_dir, "model_name.txt"), "w") as f:
        f.write(model_name)

    return index, model


def main():
    ap = argparse.ArgumentParser(description="chunks → embedding → FAISS")
    ap.add_argument("chunks_dir", help="*_chunks.jsonl 所在目录")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5",
                    help="sentence-transformers 模型名")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 chunks_dir 下的 vector_db/）")
    args = ap.parse_args()

    chunks_dir = args.chunks_dir.rstrip("/")
    out_dir = args.out or os.path.join(chunks_dir, "vector_db")

    chunks = load_all_chunks(chunks_dir)
    print(f"loaded {len(chunks)} searchable chunks from {chunks_dir}")

    build_index(chunks, args.model, out_dir)

    # 快速自测
    from sentence_transformers import SentenceTransformer
    import faiss
    model = SentenceTransformer(args.model)
    q = model.encode(["金融租赁公司注册资本最低限额是多少"], normalize_embeddings=True)
    import faiss
    idx = faiss.read_index(os.path.join(out_dir, "faiss.index"))
    D, I = idx.search(np.array(q).astype("float32"), 3)
    print("\n── 自测: '金融租赁公司注册资本最低限额是多少' ──")
    for score, chunk_id in zip(D[0], I[0]):
        print(f"  [{score:.4f}] {chunks[chunk_id]['chunk_id']}: {chunks[chunk_id]['text'][:80]}...")


if __name__ == "__main__":
    main()