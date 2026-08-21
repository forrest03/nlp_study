import sys
sys.path.insert(0, "src")
from rag_annual_report.src.rag_pipeline   import RAGPipeline

# 初始化（首次约需 10 秒：加载索引 + 构建 BM25）
pipeline = RAGPipeline(
    use_bm25=True,
    use_rerank=False,         # Rerank 需要本地模型
    use_query_rewrite=False,
)

# 单次查询
result = pipeline.query("政府可以随意征用我的土地吗？")
print(result["answer"])
print(result["citations"])     # 来源列表
print(len(result["retrieved"])) # 实际使用的 chunk 数

# 带过滤条件
result = pipeline.query(
    "具体条款",
    filter_meta={"raw_name": "中华人民共和国宪法"},
)