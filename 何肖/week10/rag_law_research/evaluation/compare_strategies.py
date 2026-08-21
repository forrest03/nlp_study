"""
消融实验脚本：对比不同分块策略和检索方式的效果

实验矩阵：
  分块策略: fixed / semantic / hierarchical
  检索方式: vector_only / bm25_only / hybrid


使用方式：
  python compare_strategies.py
  python compare_strategies.py --strategies semantic,hierarchical
  python compare_strategies.py --modes vector_only,hybrid

注意：
  运行前需要已分别建好三种策略的向量索引：
    data/chunks/all_fixed.json      → vectorstore/faiss_fixed/
    data/chunks/all_semantic.json   → vectorstore/faiss_semantic/
    data/chunks/all_hierarchical.json → vectorstore/faiss_hierarchical/

依赖：
  pip install faiss-cpu openai jieba rank_bm25
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent
EVAL_DIR      = Path(__file__).parent
RESULT_DIR    = EVAL_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL   = "text-embedding-v3"
EMBED_DIM     = 1024

# 消融实验用的 10 道题（取 questions.json 中有明确 target_docs 的题）
ABLATION_QUESTION_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 15]


# ── 加载题集 ──────────────────────────────────────────────────────────────────

def load_ablation_questions() -> list[dict]:
    qpath = EVAL_DIR / "questions.json"
    with open(qpath, encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data["questions"] if q["id"] in ABLATION_QUESTION_IDS]


# ── 动态加载指定策略的索引 ────────────────────────────────────────────────────

def load_index(strategy: str):
    """
    根据策略名加载对应的 FAISS 索引和 meta。
    索引路径约定：
      vectorstore/faiss_{strategy}/index.bin
      vectorstore/faiss_{strategy}/meta.json
    """
    import faiss
    vs_dir     = BASE_DIR / "vectorstore" / f"faiss_{strategy}"
    index_path = vs_dir / "faiss_index.bin"
    meta_path  = vs_dir / "faiss_meta.json"

    # 兼容默认路径（semantic 策略即主路径）
    if not index_path.exists():
        index_path = BASE_DIR / "vectorstore/faiss_semantic/faiss_index.bin"
        meta_path  = BASE_DIR / "vectorstore/faiss_semantic/faiss_meta.json"
        logger.warning(f"未找到 {vs_dir}，使用默认索引路径（{strategy} 策略）")

    if not index_path.exists():
        raise FileNotFoundError(
            f"找不到索引文件 {index_path}\n"
            "请先分别构建各策略的索引（见脚本顶部说明）"
        )

    index     = faiss.read_index(str(index_path))
    with open(meta_path, encoding="utf-8") as f:
        meta_list = json.load(f)
    return index, meta_list


def validate_questions_against_corpus(questions: list[dict], meta_list: list[dict]) -> float:
    """
    题库-语料一致性快速检查：
    统计有 target_docs 的题中，有多少题能在当前语料的文本/section/source_file 中找到至少一个关键词片段。
    返回覆盖率（0~1）。
    """
    def _normalize(s: str) -> str:
        if not s:
            return ""
        return (
            str(s)
            .replace("（", "(")
            .replace("）", ")")
            .replace("《", "")
            .replace("》", "")
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .lower()
        )

    corpus = " ".join(
        _normalize(" ".join([
            m.get("content", ""),
            m.get("section", ""),
            m.get("source_file", ""),
        ]))
        for m in meta_list[: min(len(meta_list), 3000)]  # 抽样足够判断域是否一致
    )

    total = 0
    covered = 0
    for q in questions:
        tds = q.get("target_docs", [])
        if not tds:
            continue
        total += 1
        hit = False
        for td in tds:
            t = _normalize(td)
            if not t:
                continue
            # 只取前 20 个字符做粗匹配，减少 target_docs 很长导致的误判
            if t[:20] and t[:20] in corpus:
                hit = True
                break
        if hit:
            covered += 1

    return (covered / total) if total else 0.0


# ── Embedding 查询 ────────────────────────────────────────────────────────────

_embed_client = None

def get_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import OpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
        _embed_client = OpenAI(
            api_key=api_key,
            base_url=DASHSCOPE_URL,
        )
    return _embed_client


def embed_query(query: str) -> np.ndarray:
    client = get_embed_client()
    resp   = client.embeddings.create(model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM)
    vec    = np.array([resp.data[0].embedding], dtype="float32")
    vec    = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    return vec


# ── 三种检索方式 ──────────────────────────────────────────────────────────────

def retrieve_vector(query: str, index, meta_list: list, top_k: int = 10) -> list[dict]:
    q_vec          = embed_query(query)
    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0: continue
        item = dict(meta_list[idx])
        item["_score"] = float(score)
        results.append(item)
    return results


def retrieve_bm25(query: str, meta_list: list, top_k: int = 10) -> list[dict]:
    import jieba
    from rank_bm25 import BM25Okapi
    # 懒加载 BM25（per-call，仅用于消融对比，不缓存）
    tokenized = [list(jieba.cut(m["content"])) for m in meta_list]
    bm25      = BM25Okapi(tokenized)
    tokens    = list(jieba.cut(query))
    scores    = bm25.get_scores(tokens)
    top_idx   = np.argsort(scores)[::-1][:top_k]
    results   = []
    for idx in top_idx:
        if scores[idx] < 1e-9: continue
        item = dict(meta_list[idx])
        item["_score"] = float(scores[idx])
        results.append(item)
    return results


def retrieve_hybrid(query: str, index, meta_list: list, top_k: int = 10) -> list[dict]:
    """RRF 混合检索。"""
    vec_res  = retrieve_vector(query, index, meta_list, top_k)
    bm25_res = retrieve_bm25(query, meta_list, top_k)

    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}
    k = 60

    for rank, item in enumerate(vec_res, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    for rank, item in enumerate(bm25_res, 1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid]  = item

    sorted_cids = sorted(rrf_scores, key=lambda x: -rrf_scores[x])
    return [chunk_map[cid] for cid in sorted_cids[:top_k]]


# ── 检索指标计算 ──────────────────────────────────────────────────────────────

def compute_metrics(
    questions: list[dict],
    index,
    meta_list: list,
    retrieval_mode: str,
    top_k: int = 4,
) -> dict:
    """
    计算：
      hit_rate@k：命中率（目标文档信息在 top-k 的 chunk 中出现）
      mrr@k：平均倒数排名
    """
    def _normalize(s: str) -> str:
        if not s:
            return ""
        return (
            str(s)
            .replace("（", "(")
            .replace("）", ")")
            .replace("《", "")
            .replace("》", "")
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .lower()
        )

    def _is_hit(item: dict, target_docs: list[str]) -> bool:
        # questions.json 的 target_docs 是文档/章节描述文本，不是 stock_code_year；
        # 需要和 chunk 的多个文本字段做模糊包含匹配。
        haystack_parts = [
            item.get("content", ""),
            item.get("section", ""),
            item.get("source_file", ""),
            item.get("chunk_id", ""),
            item.get("stock_code", ""),
            item.get("year", ""),
        ]
        haystack = " ".join(str(x) for x in haystack_parts if x)
        hay = _normalize(haystack)
        if not hay:
            return False
        for td in target_docs:
            t = _normalize(td)
            if not t:
                continue
            # 规则1：双向包含，兼容 target_docs 长句和 chunk 字段短词的差异
            if t in hay or hay in t:
                return True
            # 规则2：分词命中（处理“民法典_婚姻家庭编”“刑法 · 总则 > ...”这类描述）
            for sep in ["·", ">", "_", "，", "。", ",", ".", "(", ")", "（", "）", "：", ":"]:
                td = td.replace(sep, " ")
            tokens = [_normalize(x) for x in td.split() if _normalize(x)]
            # 去掉太短 token，避免“第”“编”等噪声词造成误判
            tokens = [x for x in tokens if len(x) >= 2]
            if not tokens:
                continue
            matched = sum(1 for tok in tokens if tok in hay)
            # 命中阈值：至少命中 2 个 token，或 token 总数较少时命中 1 个
            need = 1 if len(tokens) <= 2 else 2
            if matched >= need:
                return True
        return False

    hits = []
    mrrs = []

    for q in questions:
        target_docs = q.get("target_docs", [])
        if not target_docs:
            continue  # should_refuse 类跳过

        if retrieval_mode == "vector_only":
            retrieved = retrieve_vector(q["question"], index, meta_list, top_k * 2)
        elif retrieval_mode == "bm25_only":
            retrieved = retrieve_bm25(q["question"], meta_list, top_k * 2)
        else:
            retrieved = retrieve_hybrid(q["question"], index, meta_list, top_k * 2)

        retrieved = retrieved[:top_k]

        # 判断命中：retrieved 的文本/元数据是否覆盖 target_docs
        hit_any = False
        first_rank = None
        for i, item in enumerate(retrieved, 1):
            if _is_hit(item, target_docs):
                hit_any = True
                if first_rank is None:
                    first_rank = i

        hits.append(1 if hit_any else 0)
        mrrs.append(1 / first_rank if first_rank else 0)

    return {
        "hit_rate":  np.mean(hits) if hits else 0.0,
        "mrr":       np.mean(mrrs) if mrrs else 0.0,
        "n_questions": len(hits),
    }


# ── 主实验循环 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 消融实验")
    parser.add_argument("--strategies", type=str, default="semantic",
                        help="逗号分隔的策略列表: fixed,semantic,hierarchical")
    parser.add_argument("--modes", type=str, default="vector_only,hybrid",
                        help="逗号分隔的检索方式: vector_only,bm25_only,hybrid")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]
    modes      = [m.strip() for m in args.modes.split(",")]
    questions  = load_ablation_questions()
    logger.info(f"消融实验: {len(questions)} 道题  策略={strategies}  检索方式={modes}")

    all_results = []

    for strategy in strategies:
        logger.info(f"\n── 加载策略: {strategy} ──")
        try:
            index, meta_list = load_index(strategy)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        coverage = validate_questions_against_corpus(questions, meta_list)
        if coverage < 0.2:
            logger.warning(
                "题库与当前语料匹配度很低（覆盖率 %.0f%%），Hit/MRR 可能接近 0。\n"
                "请检查 evaluation/questions.json 是否为当前项目语料（年报）对应题集。",
                coverage * 100,
            )

        for mode in modes:
            logger.info(f"  检索方式: {mode}")
            metrics = compute_metrics(questions, index, meta_list, mode, args.top_k)
            result  = {
                "strategy":     strategy,
                "retrieval_mode": mode,
                "hit_rate":     metrics["hit_rate"],
                "mrr":          metrics["mrr"],
                "n_questions":  metrics["n_questions"],
            }
            all_results.append(result)
            logger.info(f"    Hit@{args.top_k}={metrics['hit_rate']:.3f}  MRR={metrics['mrr']:.3f}")

    # 输出汇总表格
    if all_results:
        print(f"\n{'='*70}")
        print(f"消融实验结果汇总（Top-{args.top_k}）")
        print(f"{'='*70}")
        print(f"{'分块策略':<15} {'检索方式':<15} {'Hit Rate':>10} {'MRR':>10} {'题数':>6}")
        print(f"{'-'*70}")
        for r in sorted(all_results, key=lambda x: -x["hit_rate"]):
            print(f"{r['strategy']:<15} {r['retrieval_mode']:<15} "
                  f"{r['hit_rate']:>10.3f} {r['mrr']:>10.3f} {r['n_questions']:>6}")
        print(f"{'='*70}")

        # 保存
        out_path = RESULT_DIR / "ablation_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
