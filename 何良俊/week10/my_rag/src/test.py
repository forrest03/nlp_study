"""批量测试：随机生成 20 个不重复问题，逐个问答，结果存到 output/qa_results.json。

运行：python test.py
产物：../output/qa_results.json
"""
import json
import random
import sys
from datetime import datetime

import config as C
from ask import answer_query, load_index

# 公司列表（股票代码 + 简称，用于构造问题）
COMPANIES = [
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("601318", "中国平安"),
    ("300750", "宁德时代"),
    ("002415", "海康威视"),
]
YEARS = ["2021", "2022", "2023"]

# 问题模板：{C}=公司名 {Y}=年份。覆盖营收/利润/研发/分红/风险等常见财务维度
QUESTION_TEMPLATES = [
    "{C}{Y}年的营业收入是多少？",
    "{C}{Y}年的归属于上市公司股东的净利润是多少？",
    "{C}{Y}年的研发投入金额是多少？占营业收入比例多少？",
    "{C}{Y}年的毛利率是多少？相比上年有何变化？",
    "{C}{Y}年的现金分红方案是怎样的？每10股派多少？",
    "{C}{Y}年末的总资产是多少？",
    "{C}{Y}年的经营活动产生的现金流量净额是多少？",
    "{C}{Y}年的基本每股收益是多少？",
    "{C}{Y}年公司前五大客户合计销售金额占营业收入比例是多少？",
    "{C}{Y}年公司主营业务收入构成中占比最大的业务板块是哪个？金额多少？",
    "{C}{Y}年公司资产负债率是多少？",
    "{C}{Y}年公司员工总数是多少？相比上年有何变化？",
    "{C}{Y}年公司提到的主要经营风险有哪些？",
    "{C}{Y}年公司的净资产收益率(ROE)是多少？",
    "{C}{Y}年公司在研发人员数量是多少？",
    "{C}{Y}年公司的应收账款余额是多少？",
    "{C}{Y}年公司的存货账面余额是多少？",
    "{C}{Y}年公司的第一大供应商采购额占采购总额比例是多少？",
    "{C}{Y}年公司的所得税费用是多少？",
    "{C}{Y}年公司管理层对行业未来发展的判断是什么？",
    "{C}{Y}年公司投资活动产生的现金流量净额是多少？",
    "{C}{Y}年公司的营业成本是多少？",
    "{C}{Y}年公司的销售费用是多少？",
    "{C}{Y}年公司支付给职工以及为职工支付的现金是多少？",
]


def generate_questions(n=20, seed=42):
    """从公司×年份×模板组合中随机抽 n 个不重复问题。"""
    pool = []
    for code, name in COMPANIES:
        for y in YEARS:
            for tpl in QUESTION_TEMPLATES:
                pool.append(tpl.format(C=name, Y=y))
    rng = random.Random(seed)
    n = min(n, len(pool))
    return rng.sample(pool, n)


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

    questions = generate_questions(20)
    print(f"[生成] {len(questions)} 个问题")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print("\n[加载] 索引（FAISS + BM25）...")
    index, chunks, bm25 = load_index()

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q}")
        res = answer_query(q, index, chunks, bm25)
        if res["error"]:
            print(f"    ✗ {res['error']}")
        else:
            srcs = ", ".join(
                f"[{s['idx']}]{s['company']}{s['year']}p{s['page_start']}"
                for s in res["sources"]
            )
            print(f"    ✓ 来源: {srcs}")
        results.append(res)

    # 落盘到 output/
    out_dir = C.ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qa_results.json"
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": C.LLM_MODEL,
        "embed_model": C.EMBED_MODEL,
        "retrieval": "hybrid (dense + BM25 + RRF)",
        "top_k": C.TOP_K,
        "candidate_k": C.CANDIDATE_K,
        "rrf_k": C.RRF_K,
        "total": len(results),
        "succeeded": sum(1 for r in results if not r["error"]),
        "failed": sum(1 for r in results if r["error"]),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 成功 {payload['succeeded']} / 失败 {payload['failed']}")
    print(f"[落盘] {out_path}")


if __name__ == "__main__":
    main()
