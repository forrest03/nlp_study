"""构建 FAISS 索引：从 raw_pdf 原始 PDF 抽取文本 → 自行切分 → DashScope 嵌入 → 保存索引。

运行：python build_index.py
产物：index/faiss.index、index/chunks.json
"""
import json
import re
import sys
import time

import fitz  # PyMuPDF
import numpy as np
import requests

import config as C

# 句子切分：在中文/英文句末标点后断开，保留标点
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
# 需过滤的噪声行：纯页码（如 "1 / 143"）、纯点号目录线（"...."）
_NOISE_LINE = re.compile(r"^[\s\d/页P\-—.·、，,]+$")


# ============ 切分 ============
def load_manifest():
    """读取 manifest.json，返回每份报告的元数据 + PDF 绝对路径。"""
    with open(C.MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    items = []
    for m in manifest:
        pdf = C.RAW_PDF_DIR / m["filename"]
        if not pdf.exists():
            print(f"[警告] PDF 不存在，跳过：{pdf}")
            continue
        items.append({**m, "pdf_path": pdf})
    return items


def extract_sentences(pdf_path):
    """用 PyMuPDF 逐页抽取文本，切分为带页码的句子列表。返回 [(sentence, page_num), ...]"""
    doc = fitz.open(pdf_path)
    sent_pages = []
    for page in doc:
        page_no = page.number + 1
        text = page.get_text("text") or ""
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and not _NOISE_LINE.match(ln)]
        text = " ".join(lines)
        for s in _SENT_SPLIT.split(text):
            s = s.strip()
            if len(s) >= 2:
                sent_pages.append((s, page_no))
    doc.close()
    return sent_pages


def chunk_sentences(sent_pages, meta):
    """把句子按目标长度贪心合并成 chunk，相邻 chunk 保留一定重叠。"""
    chunks = []
    buf = []  # 存 (sentence, page) 元组
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return None
        content = "".join(s for s, _ in buf)
        chunk = {
            "content": content,
            "metadata": {
                "stock_code": meta["stock_code"],
                "company_name": meta["company_name"],
                "year": meta["year"],
                "source_file": meta["filename"],
                "page_start": buf[0][1],
                "page_end": buf[-1][1],
            },
        }
        # 保留尾部若干句子作为下一 chunk 的重叠开头
        overlap, overlap_len = [], 0
        for sp in reversed(buf):
            if overlap_len >= C.CHUNK_OVERLAP:
                break
            overlap.append(sp)
            overlap_len += len(sp[0])
        overlap.reverse()
        buf = overlap
        buf_len = sum(len(s) for s, _ in buf)
        return chunk

    for s, p in sent_pages:
        if len(s) > C.CHUNK_SIZE * 2:
            s = s[: C.CHUNK_SIZE * 2]
        if buf and buf_len + len(s) > C.CHUNK_SIZE:
            chunk = flush()
            if chunk:
                chunks.append(chunk)
        buf.append((s, p))
        buf_len += len(s)
    chunk = flush()
    if chunk:
        chunks.append(chunk)

    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{meta['stock_code']}_{meta['year']}_{i:05d}"
    return chunks


def build_all_chunks():
    """遍历所有 PDF，生成全部 chunks。"""
    items = load_manifest()
    all_chunks = []
    for it in items:
        print(f"[切分] {it['company_name']} {it['year']} ...")
        sents = extract_sentences(it["pdf_path"])
        chunks = chunk_sentences(sents, it)
        print(f"        句子 {len(sents)} → 片段 {len(chunks)}")
        all_chunks.extend(chunks)
    return all_chunks


# ============ BM25 分词 ============
def tokenize(text):
    """jieba 分词，返回词列表（用于 BM25 稀疏检索）。"""
    import jieba

    return [t for t in jieba.lcut(text) if t.strip()]


def build_bm25_corpus(chunks):
    """对全部 chunk 分词，返回 tokenized corpus（list[list[str]]）。"""
    print("[BM25] 分词构建语料 ...")
    corpus = []
    for c in chunks:
        corpus.append(tokenize(c["content"]))
    return corpus


# ============ 嵌入（DashScope API）============
def _post_embeddings(inputs):
    """调用 DashScope /embeddings，返回与 inputs 等长的向量列表。"""
    url = f"{C.EMBED_BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {C.EMBED_API_KEY}"}
    data = {"model": C.EMBED_MODEL, "input": inputs, "encoding_format": "float"}
    resp = requests.post(url, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    objs = resp.json()["data"]
    objs.sort(key=lambda o: o["index"])
    return [o["embedding"] for o in objs]


def _embed_batch_robust(batch, depth=0):
    """带重试与失败降级切分的批量嵌入。"""
    last_err = None
    for attempt in range(4):
        try:
            return _post_embeddings(batch)
        except Exception as e:  # 网络抖动 / 限流
            last_err = e
            time.sleep(1.5 ** attempt)
    if depth < 4 and len(batch) > 1:  # 仍失败则对半切分重试
        mid = len(batch) // 2
        return _embed_batch_robust(batch[:mid], depth + 1) + \
               _embed_batch_robust(batch[mid:], depth + 1)
    raise RuntimeError(f"嵌入失败（已重试/切分）：{last_err}")


def _normalize(vecs):
    """L2 归一化，便于用内积近似余弦相似度。"""
    arr = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_texts(texts, desc="嵌入"):
    """批量嵌入多段文本，返回 (N, dim) 的归一化 float32 矩阵。"""
    if not C.EMBED_API_KEY:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY 环境变量")
    all_vecs = []
    n = len(texts)
    for i in range(0, n, C.EMBED_BATCH):
        batch = texts[i : i + C.EMBED_BATCH]
        vecs = _embed_batch_robust(batch)
        all_vecs.extend(vecs)
        print(f"\r[{desc}] {min(i + C.EMBED_BATCH, n)}/{n}", end="", flush=True)
    print()
    return _normalize(all_vecs)


def embed_query(query):
    """嵌入单条 query，返回 (1, dim) 归一化向量。"""
    return _normalize([_post_embeddings([query])[0]])


# ============ 主流程 ============
def build_index():
    import faiss

    if not C.EMBED_API_KEY:
        print("错误：未设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)

    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = build_all_chunks()
    print(f"\n[总计] 片段数：{len(chunks)}")

    texts = [c["content"] for c in chunks]
    print(f"[嵌入] 调用 DashScope {C.EMBED_MODEL} ...")
    emb = embed_texts(texts, desc="嵌入")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    print(f"[索引] FAISS 构建完成，维度 {dim}，向量数 {index.ntotal}")

    faiss.write_index(index, str(C.INDEX_DIR / "faiss.index"))
    with open(C.INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    # BM25 分词语料（运行时据此重建 BM25Okapi 对象，很快）
    bm25_corpus = build_bm25_corpus(chunks)
    with open(C.INDEX_DIR / "bm25_corpus.json", "w", encoding="utf-8") as f:
        json.dump(bm25_corpus, f, ensure_ascii=False)

    print(f"[落盘] 向量索引 → {C.INDEX_DIR / 'faiss.index'}")
    print(f"[落盘] 片段元数据 → {C.INDEX_DIR / 'chunks.json'}")
    print(f"[落盘] BM25 语料 → {C.INDEX_DIR / 'bm25_corpus.json'}")


if __name__ == "__main__":
    sys.exit(build_index())
