"""
RAG 评估脚本

评估指标：
  - 答案长度
  - 引用数量
  - 检索到的上下文数量
  - 回答时间

使用方式：
  python evaluate.py --pipeline native
  python evaluate.py --pipeline langchain
  python evaluate.py --pipeline both
  python evaluate.py --pipeline native --question-ids 1,2,3

注意：RAGAS 评估因依赖问题暂不支持，如需完整评估请手动安装并配置环境。
"""

import os
import json
import time
import argparse
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).parent.parent
TEST_FILE = BASE_DIR / "resource" / "test_questions.json"

DEFAULT_QUESTIONS = [
    {
        "id": 1,
        "question": "2025年重点领域和行业节能改造节能量多少标准煤？",
        "ground_truth": "节能降碳改造形成节能量约5000万吨标准煤",
    },
    {
        "id": 2,
        "question": "节能降碳行动方案的总体要求是什么？",
        "ground_truth": "深入贯彻落实党中央、国务院关于碳达峰碳中和决策部署，坚持节约优先、源头减量、科技支撑、制度保障",
    },
    {
        "id": 3,
        "question": "产品碳足迹核算标准的编制原则是什么？",
        "ground_truth": "科学性、规范性、适用性、可操作性",
    },
    {
        "id": 4,
        "question": "2024年节能降碳的重点任务有哪些？",
        "ground_truth": "重点领域节能降碳改造、园区节能降碳改造、城镇节能降碳改造",
    },
    {
        "id": 5,
        "question": "节能降碳改造完成后预计减排二氧化碳多少？",
        "ground_truth": "减排二氧化碳约1.3亿吨",
    },
    {
        "id": 6,
        "question": "碳足迹核算的边界包括哪些？",
        "ground_truth": "原材料获取、生产制造、运输仓储、使用废弃等全生命周期阶段",
    },
    {
        "id": 7,
        "question": "节能降碳行动方案中提到的重点行业有哪些？",
        "ground_truth": "钢铁、有色、建材、石化、化工等传统高耗能行业",
    },
    {
        "id": 8,
        "question": "产品碳足迹标准适用于哪些产品？",
        "ground_truth": "各类工业产品、农产品、服务等",
    },
    {
        "id": 9,
        "question": "节能降碳行动方案的实施期限是多久？",
        "ground_truth": "2024—2025年",
    },
    {
        "id": 10,
        "question": "碳足迹核算需要收集哪些数据？",
        "ground_truth": "能源消耗数据、原材料消耗数据、排放因子数据等",
    },
]


def ensure_test_file():
    if not TEST_FILE.exists():
        TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_QUESTIONS, f, ensure_ascii=False, indent=2)
        print(f"测试文件已创建: {TEST_FILE}")
    return TEST_FILE


def load_test_questions(question_ids: Optional[List[int]] = None) -> List[Dict]:
    with open(ensure_test_file(), encoding="utf-8") as f:
        questions = json.load(f)
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    return questions


def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(a.replace(" ", ""))
    b_tokens = set(b.replace(" ", ""))
    if not a_tokens and not b_tokens:
        return 1.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union > 0 else 0.0


class PipelineRunner:
    def __init__(self, pipeline_type: str, api_key: str):
        self.type = pipeline_type
        self.api_key = api_key
        self.pipeline = None

    def init(self):
        if self.type == "native":
            spec = importlib.util.spec_from_file_location(
                "rag_pipeline",
                Path(__file__).parent / "rag_pipeline.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.pipeline = module.RAGPipeline(
                use_bm25=True,
                use_rerank=False,
                use_query_rewrite=False,
            )
        elif self.type == "langchain":
            spec = importlib.util.spec_from_file_location(
                "rag_pipeline_langchain",
                Path(__file__).parent / "rag_pipeline_langchain.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.pipeline = module.LangChainRAGPipeline(
                api_key=self.api_key,
                use_bm25=True,
                use_rerank=False,
                use_query_rewrite=False,
            )
        else:
            raise ValueError(f"Unknown pipeline type: {self.type}")

    def query(self, question: str) -> Dict:
        if self.type == "native":
            return self.pipeline.query(question, filter_meta=None, verbose=False)
        elif self.type == "langchain":
            return self.pipeline.query(question, years=None, verbose=False)


def run_evaluation(pipeline_type: str, questions: List[Dict]) -> Dict:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")

    runner = PipelineRunner(pipeline_type, api_key)
    runner.init()

    results = []
    total_time = 0
    total_similarity = 0
    success_count = 0

    print(f"\n{'='*60}")
    print(f"评估 {pipeline_type} 版，共 {len(questions)} 题")
    print(f"{'='*60}")

    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['question']}")

        start = time.time()
        try:
            result = runner.query(q["question"])
            elapsed = time.time() - start
            total_time += elapsed

            answer = result["answer"]
            citations = result.get("citations", [])

            retrieved_content = []
            if "retrieved" in result:
                for item in result["retrieved"]:
                    if isinstance(item, dict):
                        retrieved_content.append(item.get("content", ""))
                    elif hasattr(item, "page_content"):
                        retrieved_content.append(item.page_content)

            similarity = jaccard_similarity(answer, q.get("ground_truth", ""))
            total_similarity += similarity

            success_count += 1

            print(f"  耗时: {elapsed:.2f}s")
            print(f"  相似度: {similarity:.4f}")
            print(f"  引用数: {len(citations)}")
            print(f"  答案: {answer[:120]}..." if len(answer) > 120 else f"  答案: {answer}")

            results.append({
                "id": q["id"],
                "question": q["question"],
                "answer": answer,
                "ground_truth": q.get("ground_truth", ""),
                "similarity": similarity,
                "citation_count": len(citations),
                "retrieved_count": len(retrieved_content),
                "time": elapsed,
            })
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results.append({
                "id": q["id"],
                "question": q["question"],
                "answer": "",
                "ground_truth": q.get("ground_truth", ""),
                "similarity": 0,
                "citation_count": 0,
                "retrieved_count": 0,
                "time": 0,
                "error": str(e),
            })

    avg_time = total_time / len(questions) if questions else 0
    avg_similarity = total_similarity / success_count if success_count > 0 else 0

    metrics = {
        "success_rate": success_count / len(questions) if questions else 0,
        "avg_time": avg_time,
        "avg_similarity": avg_similarity,
        "avg_citations": sum(r.get("citation_count", 0) for r in results) / len(results) if results else 0,
        "avg_retrieved": sum(r.get("retrieved_count", 0) for r in results) / len(results) if results else 0,
    }

    return {
        "pipeline": pipeline_type,
        "total_questions": len(questions),
        "success_count": success_count,
        "metrics": metrics,
        "results": results,
    }


def print_metrics_summary(result: Dict):
    m = result["metrics"]
    print("\n" + "─" * 60)
    print(f"{result['pipeline']} 版评估指标汇总")
    print("─" * 60)
    print(f"  成功率        : {m['success_rate']:.2%}")
    print(f"  平均耗时      : {m['avg_time']:.2f}s")
    print(f"  平均相似度    : {m['avg_similarity']:.4f}")
    print(f"  平均引用数    : {m['avg_citations']:.2f}")
    print(f"  平均检索数    : {m['avg_retrieved']:.2f}")


def print_comparison_table(native_result: Dict, langchain_result: Dict):
    print("\n" + "=" * 70)
    print(f"{'指标':<20} {'Native 版':<15} {'LangChain 版':<15} {'差异'}")
    print("=" * 70)

    def get_metric(r, name):
        m = r.get("metrics", {})
        return m.get(name, 0.0)

    metrics = [
        ("success_rate", "成功率"),
        ("avg_time", "平均耗时(s)"),
        ("avg_similarity", "平均相似度"),
        ("avg_citations", "平均引用数"),
        ("avg_retrieved", "平均检索数"),
    ]

    for key, name in metrics:
        native_val = get_metric(native_result, key)
        langchain_val = get_metric(langchain_result, key)
        if key == "avg_time":
            diff = langchain_val - native_val
        else:
            diff = langchain_val - native_val
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        if key == "success_rate":
            print(f"{name:<20} {native_val:<15.2%} {langchain_val:<15.2%} {diff_str}")
        else:
            print(f"{name:<20} {native_val:<15.4f} {langchain_val:<15.4f} {diff_str}")


def main():
    parser = argparse.ArgumentParser(description="RAG 评估脚本")
    parser.add_argument("--pipeline", type=str, required=True,
                        choices=["native", "langchain", "both"],
                        help="评估哪个版本")
    parser.add_argument("--question-ids", type=str, default=None,
                        help="指定题目ID，逗号分隔")
    args = parser.parse_args()

    question_ids = None
    if args.question_ids:
        question_ids = [int(x.strip()) for x in args.question_ids.split(",")]

    questions = load_test_questions(question_ids)
    print(f"加载 {len(questions)} 道测试题")

    results = {}

    if args.pipeline in ["native", "both"]:
        native_result = run_evaluation("native", questions)
        results["native"] = native_result
        print_metrics_summary(native_result)

    if args.pipeline in ["langchain", "both"]:
        langchain_result = run_evaluation("langchain", questions)
        results["langchain"] = langchain_result
        print_metrics_summary(langchain_result)

    if args.pipeline == "both" and "native" in results and "langchain" in results:
        print_comparison_table(results["native"], results["langchain"])

    output_file = BASE_DIR / "resource" / f"evaluation_result_{args.pipeline}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存: {output_file}")


if __name__ == "__main__":
    main()