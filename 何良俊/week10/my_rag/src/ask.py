"""交互式问答：加载 FAISS 索引 → 检索相关片段 → DeepSeek 生成回答。

运行：python ask.py
"""
import json
import sys

import faiss
import requests

import config as C
from build_index import embed_query

# 指令模板：约束模型只依据片段作答并标注来源
SYSTEM_PROMPT = (
    "你是一名财务分析师助理。请仅依据下面给出的年报片段回答用户问题。"
    "不要编造或使用片段外的信息；若片段不足以回答，请明确说明并指出需要哪类信息。"
    "回答中引用具体数据或结论时，在句末用方括号标注对应片段的编号，如 [1]、[2][3]，"
    "编号必须与下方给出的片段序号一致；不要在正文里写公司名、年份、页码等来源信息。"
    "回答使用中文，条理清晰。"
)


def load_index():
    """加载 FAISS 索引、chunks 元数据、BM25 索引。返回 (faiss_index, chunks, bm25)。"""
    from rank_bm25 import BM25Okapi
    from build_index import tokenize

    faiss_index = faiss.read_index(str(C.INDEX_DIR / "faiss.index"))
    with open(C.INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(C.INDEX_DIR / "bm25_corpus.json", encoding="utf-8") as f:
        bm25_corpus = json.load(f)
    bm25 = BM25Okapi(bm25_corpus)
    return faiss_index, chunks, bm25


def _dense_search(query, faiss_index, chunks, k):
    """向量检索：返回 [(chunk_idx, score), ...]，按相似度降序，最多 k 条。"""
    qv = embed_query(query)
    scores, idx = faiss_index.search(qv, k)
    out = []
    for sc, i in zip(scores[0], idx[0]):
        if i >= 0:
            out.append((int(i), float(sc)))
    return out


def _bm25_search(query, bm25, chunks, k):
    """BM25 检索：返回 [(chunk_idx, score), ...]，按 BM25 分降序，最多 k 条。"""
    from build_index import tokenize

    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    # 取 top k 的下标
    import numpy as np

    k = min(k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    # 按分数降序排
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


def _rrf_fuse(dense_hits, bm25_hits, k=C.RRF_K):
    """RRF 融合排名：score(d) = Σ 1/(k + rank)。

    dense_hits / bm25_hits 已按各自分数降序，rank 从 1 开始。
    返回融合后按 RRF 分降序的 [(chunk_idx, rrf_score), ...]。
    """
    scores = {}
    for rank, (idx, _) in enumerate(dense_hits, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(bm25_hits, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def retrieve(query, faiss_index, chunks, bm25, top_k=C.TOP_K):
    """混合检索：向量 + BM25 各取 CANDIDATE_K 条，RRF 融合后取 top_k。"""
    dense = _dense_search(query, faiss_index, chunks, C.CANDIDATE_K)
    bm25h = _bm25_search(query, bm25, chunks, C.CANDIDATE_K)
    fused = _rrf_fuse(dense, bm25h)
    hits = []
    for idx, rrf_score in fused[:top_k]:
        c = chunks[idx]
        hits.append({
            "score": round(rrf_score, 4),
            "content": c["content"],
            "meta": c["metadata"],
        })
    return hits


def build_context(hits):
    """把检索片段拼成带编号的上下文文本（编号即正文引用的 [i]）。"""
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}]:\n{h['content']}")
    return "\n\n".join(parts)


def generate(question, context):
    """调用 DeepSeek（OpenAI 兼容）/chat/completions 生成回答。"""
    user_msg = f"年报片段：\n{context}\n\n问题：{question}\n\n请依据上述片段回答。"
    payload = {
        "model": C.LLM_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {C.LLM_API_KEY}"}
    resp = requests.post(
        f"{C.LLM_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=300
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def answer_query(question, index, chunks, bm25):
    """对单个问题执行「检索→生成」完整流程，返回结构化结果。

    返回 dict：{question, answer, sources:[{idx, company, year, page_start,
    page_end, score, preview}], error}
    出错时 answer 为空、error 填错误信息，便于批量调用统一处理。
    """
    result = {"question": question, "answer": "", "sources": [], "error": ""}
    try:
        hits = retrieve(question, index, chunks, bm25)
    except Exception as e:
        result["error"] = f"检索失败：{e}"
        return result
    if not hits:
        result["error"] = "未检索到相关片段"
        return result

    context = build_context(hits)
    try:
        result["answer"] = generate(question, context)
    except requests.RequestException as e:
        result["error"] = f"生成失败：{e}"
        return result

    for i, h in enumerate(hits, 1):
        m = h["meta"]
        preview = h["content"].replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        result["sources"].append({
            "idx": i,
            "company": m["company_name"],
            "year": m["year"],
            "page_start": m["page_start"],
            "page_end": m["page_end"],
            "score": round(h["score"], 4),
            "preview": preview,
        })
    return result


def format_sources(result):
    """把 answer_query 的结果格式化成带提示说明的来源展示文本。"""
    lines = []
    lines.append("【来源说明】（正文中的 [i] 对应下方第 i 条；score 为向量+BM25 的 RRF 融合分，越大越相关）")
    lines.append("提示：回答仅基于检索到的片段，可能不完整；如需精确数据，请以原报告对应页为准。")
    for s in result["sources"]:
        lines.append(
            f"  [{s['idx']}] {s['company']} {s['year']} 第{s['page_start']}-{s['page_end']}页"
            f"  (RRF {s['score']:.4f})"
        )
        lines.append(f"       片段预览：{s['preview']}")
    return "\n".join(lines)


def main():
    if not C.LLM_API_KEY:
        print("错误：未设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)
    if not C.EMBED_API_KEY:
        print("错误：未设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)
    if not (C.INDEX_DIR / "faiss.index").exists():
        print("索引不存在，请先运行：python build_index.py")
        sys.exit(1)

    print("[加载] 索引（FAISS + BM25）...")
    index, chunks, bm25 = load_index()
    print(f"[就绪] 共 {len(chunks)} 个片段。输入问题开始问答，Ctrl+C 或输入 exit 退出。\n")

    while True:
        try:
            q = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("再见。")
            break

        result = answer_query(q, index, chunks, bm25)
        if result["error"]:
            print(f"[错误] {result['error']}\n")
            continue

        print(f"\n答> {result['answer']}")
        print()
        print(format_sources(result))
        print()


if __name__ == "__main__":
    main()
