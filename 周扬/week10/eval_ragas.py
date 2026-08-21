#!/usr/bin/env python3
"""
RAGAS 评估脚本 — 法规知识库检索质量评估
评估模型: DeepSeek API
"""
import sys, os, json, time
import numpy as np

# ── 配置 ──
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"   # DeepSeek-V3

DB_DIR = os.path.join(os.path.dirname(__file__), "知识库原始文件/vector_db")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "eval_report.md")


# ═══════════════════════════════════════════════
# 测试用例 — 覆盖6份文档 + 不同查询类型
# ═══════════════════════════════════════════════
TEST_CASES = [
    # ── 金融租赁公司管理办法 ──
    {
        "question": "金融租赁公司的注册资本最低限额是多少？",
        "ground_truth": "注册资本为一次性实缴货币资本，最低限额为10亿元人民币或等值的可自由兑换货币。"
    },
    {
        "question": "金融租赁公司股东有哪些义务？至少列出5项。",
        "ground_truth": "股东义务包括：使用自有资金入股、持股比例合规、如实告知财务信息、控股股东变化书面告知、合并分立等重大事项告知、股份涉诉告知、关联交易合规、不滥用股东权利、配合监管调查、主要股东5年内不得转让股权、不质押股权、必要时补充资本等。"
    },
    {
        "question": "什么是售后回租业务？",
        "ground_truth": "售后回租业务是指承租人和出卖人为同一人的融资租赁业务，即承租人将自有资产出卖给出租人，同时与出租人签订融资租赁合同，再将该资产从出租人处租回。"
    },
    {
        "question": "金融租赁公司的监管指标有哪些？",
        "ground_truth": "包括资本充足率、杠杆率（不低于6%）、财务杠杆倍数（总资产不超过净资产10倍）、同业拆借比例（不超过资本净额100%）、拨备覆盖率（不低于100%）、租赁应收款拨备率（不低于2.5%）、单一客户融资集中度（不超过30%）、单一集团客户融资集中度（不超过50%）、关联度指标、流动性指标、固定收益类投资比例（不超过20%）等13项。"
    },
    {
        "question": "金融租赁公司设立专业子公司的条件是什么？",
        "ground_truth": "需经国家金融监督管理总局批准，注册资本最低3亿元，有符合资格的董事高管和从业人员，有健全的公司治理和风控体系。金融租赁公司应100%控股或持股不低于51%。可在境内保税区、自贸区、境外设立。"
    },
    # ── 货币经纪公司管理办法 ──
    {
        "question": "货币经纪公司的设立条件是什么？",
        "ground_truth": "货币经纪公司注册资本最低限额为2000万元人民币或等值可自由兑换货币，主要出资人应为境内外依法设立的金融机构，最近2年连续盈利，具有健全的公司治理和风控制度。"
    },
    # ── 银行保险机构许可证管理办法 ──
    {
        "question": "许可证遗失后银行保险机构应该怎么做？",
        "ground_truth": "应立即报告发证机关，并于发现之日起七日内发布遗失声明公告、重新领取许可证。报告内容包括机构名称、住所、批准日期、许可证流水号、机构编码、颁发日期、当事人、时间、地点、原因、过程等。领取新许可证还需提交遗失声明公告及处理结果报告。"
    },
    # ── 银行卡清算机构管理办法 ──
    {
        "question": "银行卡清算机构的注册资本要求是多少？",
        "ground_truth": "注册资本应当不低于10亿元人民币，出资人应以自有资金出资。"
    },
    # ── 行政许可实施程序规定 ──
    {
        "question": "行政许可申请受理的审查期限是多久？",
        "ground_truth": "收到申请材料后5个工作日内完成审查。申请材料不齐全或不符合要求的，应在5个工作日内一次性告知需补正的全部内容。"
    },
    # ── 资产管理产品信息披露 ──
    {
        "question": "银行保险机构资产管理产品信息披露的原则是什么？",
        "ground_truth": "应遵循真实性、准确性、完整性、及时性和公平性原则，保障投资者的知情权。"
    },
    # ── 跨文档查询 ──
    {
        "question": "金融租赁公司和银行卡清算机构的注册资本最低要求分别是多少？",
        "ground_truth": "金融租赁公司注册资本最低10亿元，银行卡清算机构注册资本最低10亿元。"
    },
    {
        "question": "金融租赁公司的关联交易如何管理？",
        "ground_truth": "应加强关联交易管理，制定管理制度，明确审批程序和标准。关联交易以商业原则进行，不优于非关联方同类交易条件。重大关联交易（单笔达上季末资本净额5%以上或累计10%以上）需经董事会批准。"
    },
]


def load_retriever():
    """加载 FAISS 索引 + embedding 模型"""
    import faiss, warnings
    warnings.filterwarnings("ignore")
    from sentence_transformers import SentenceTransformer

    model_name = open(os.path.join(DB_DIR, "model_name.txt")).read().strip()
    model = SentenceTransformer(model_name)
    index = faiss.read_index(os.path.join(DB_DIR, "faiss.index"))
    meta = json.load(open(os.path.join(DB_DIR, "meta.json")))
    return model, index, meta


def retrieve(model, index, meta, query: str, top_k: int = 5) -> list:
    """检索 top_k 个相关 chunk"""
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_vec, top_k)
    return [meta[idx]["text"] for idx in I[0]]


def llm_generate(prompt: str, max_tokens: int = 1024) -> str:
    """调用 DeepSeek API 生成回答"""
    from openai import OpenAI

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  LLM 调用失败 (attempt {attempt+1}): {e}")
            time.sleep(2 * (attempt + 1))
    return "[ERROR: LLM 调用失败]"


def build_answer(query: str, contexts: list) -> str:
    """基于检索到的上下文生成答案"""
    ctx_text = "\n\n---\n\n".join(contexts)
    prompt = f"""你是一个中国金融法规专家。请基于以下法规条文回答问题。如果上下文中没有足够信息，请诚实说明。

【法规原文】
{ctx_text}

【问题】
{query}

【要求】
1. 仅依据上述法规原文回答，不要编造内容
2. 回答简洁准确，引用具体条款编号
3. 如果法规原文不足以完整回答，说明哪些信息缺失"""
    return llm_generate(prompt)


def run_evaluation():
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("  RAGAS 法规知识库评估")
    print("=" * 60)

    # 1. 加载检索器
    print("\n[1/4] 加载检索器...")
    model, index, meta = load_retriever()
    print(f"  向量库: {index.ntotal} vectors, {len(meta)} chunks")

    # 2. 运行测试
    print(f"\n[2/4] 运行 {len(TEST_CASES)} 个测试用例...")
    results = []
    for i, tc in enumerate(TEST_CASES):
        q = tc["question"]
        gt = tc["ground_truth"]
        print(f"\n  [{i+1}/{len(TEST_CASES)}] {q[:50]}...")

        # 检索
        contexts = retrieve(model, index, meta, q, top_k=5)
        print(f"      检索: {len(contexts)} chunks")

        # 生成答案
        answer = build_answer(q, contexts)
        print(f"      生成: {len(answer)} chars")

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "contexts": contexts,
        })

    # 3. RAGAS 评估
    print(f"\n[3/4] RAGAS 评估...")
    metrics = compute_ragas_metrics(results)

    # 4. 生成报告
    print(f"\n[4/4] 生成报告 → {REPORT_PATH}")
    generate_report(results, metrics, REPORT_PATH)

    print(f"\n完成！报告: {REPORT_PATH}")
    return results, metrics


def compute_ragas_metrics(results: list) -> dict:
    """计算 RAGAS 核心指标"""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    # 用 DeepSeek 作为 ragas 的评判 LLM
    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.0,
    ))

    # 复用本地检索阶段的 embedding 模型，避免 ragas 默认回退到 OpenAI embeddings
    embed_model_name = open(os.path.join(DB_DIR, "model_name.txt")).read().strip()
    eval_embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

    # 构建样本
    samples = []
    for r in results:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            reference=r["ground_truth"],
            retrieved_contexts=r["contexts"],
        ))

    dataset = EvaluationDataset(samples=samples)

    # 计算指标
    metrics_list = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]

    scores = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # 聚合
    result = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        vals = []
        for value in scores[metric_name]:
            if value is None:
                continue
            value = float(value)
            if np.isnan(value):
                continue
            vals.append(value)
        if vals:
            result[metric_name] = {
                "mean": round(sum(vals) / len(vals), 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
                "values": [round(v, 4) for v in vals],
            }

    return result


def generate_report(results: list, metrics: dict, path: str):
    """生成 Markdown 评估报告"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append(f"# RAG 法规知识库评估报告\n")
    lines.append(f"**评估时间**: {now}")
    lines.append(f"**评估模型**: {DEEPSEEK_MODEL} (DeepSeek API)")
    lines.append(f"**Embedding模型**: BAAI/bge-small-zh-v1.5")
    lines.append(f"**测试用例数**: {len(results)}")
    lines.append(f"**知识库规模**: 6份法规, 271 chunks\n")

    # ── 指标总览 ──
    lines.append("## 一、评估指标总览\n")
    lines.append("| 指标 | 均值 | 最低 | 最高 | 说明 |")
    lines.append("|------|------|------|------|------|")
    metric_desc = {
        "faithfulness": "答案忠实度 — 生成内容是否完全基于上下文，无幻觉",
        "answer_relevancy": "答案相关性 — 答案是否切题",
        "context_precision": "上下文精确度 — 检索到的chunk是否相关",
        "context_recall": "上下文召回率 — 答案所需信息是否被检索到",
    }
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['mean']} | {m['min']} | {m['max']} | {metric_desc.get(name, '')} |")
    lines.append("")

    # ── 各用例详情 ──
    lines.append("## 二、逐用例详情\n")
    for i, r in enumerate(results):
        lines.append(f"### 用例 {i+1}: {r['question']}\n")
        lines.append(f"**参考答案**: {r['ground_truth'][:200]}...\n")
        lines.append(f"**生成答案**:")
        lines.append(f"\n{r['answer']}\n")

        # 各指标分值
        lines.append(f"**指标得分**:")
        for name in metrics:
            if i < len(metrics[name]["values"]):
                lines.append(f"- {name}: {metrics[name]['values'][i]}")
        lines.append("")

    # ── 检索命中分析 ──
    lines.append("## 三、检索命中分析\n")
    for i, r in enumerate(results):
        lines.append(f"### 用例 {i+1}: {r['question']}\n")
        for j, ctx in enumerate(r["contexts"][:3]):
            # 截取前150字符
            preview = ctx.replace("\n", " ")[:150]
            lines.append(f"- **chunk {j+1}**: {preview}...")
        lines.append("")

    # ── 改进建议 ──
    lines.append("## 四、分析与建议\n")
    if metrics:
        cp = metrics.get("context_precision", {}).get("mean", 0)
        cr = metrics.get("context_recall", {}).get("mean", 0)
        fs = metrics.get("faithfulness", {}).get("mean", 0)
        ar = metrics.get("answer_relevancy", {}).get("mean", 0)

        issues = []
        if cp < 0.7:
            issues.append("Context Precision 偏低 → 检索返回了不相关内容，建议：调整 chunk 大小、增加 reranker、优化 embedding 模型")
        if cr < 0.7:
            issues.append("Context Recall 偏低 → 关键信息未被检索到，建议：增加 top_k、使用 HyDE 查询改写、增加关键词检索")
        if fs < 0.7:
            issues.append("Faithfulness 偏低 → 生成答案包含非检索来源内容，建议：收紧 prompt 约束、降低 temperature")
        if ar < 0.7:
            issues.append("Answer Relevancy 偏低 → 答案偏离问题，建议：优化 prompt 模板")

        for issue in issues:
            lines.append(f"- ⚠️ {issue}")

        if not issues:
            lines.append("各项指标表现良好，知识库检索质量达标。")

        avg = (cp + cr + fs + ar) / 4
        lines.append(f"\n**综合评分**: {avg:.2%} (4项指标均值)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
