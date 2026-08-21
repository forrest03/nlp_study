# 法律文档 RAG 项目（`rag_law_research`）

基于《中华人民共和国刑法》《中华人民共和国宪法》《中华人民共和国民法典》构建法律问答 RAG 系统，包含：

- 原生版 Pipeline（FAISS + BM25 + RRF + 可选 Rerank）
- LangChain LCEL 版（FAISS Retriever + Prompt + LLM）
- 检索消融评测与自动评估脚本

---

## 1. 项目结构

```text
rag_law_research/
├── data/
│   ├── parsed/                 # parse_pdf.py 解析结果
│   └── chunks/                 # chunk_documents.py 分块结果
├── src/                        # 原生版 RAG
│   ├── parse_pdf.py
│   ├── chunk_documents.py
│   ├── build_index.py
│   ├── rag_pipeline.py
│   └── serve.py
├── src_langchain/              # LangChain 版 RAG
│   ├── download_model.py
│   ├── build_index_lc.py
│   └── rag_chain_lc.py
├── evaluation/
│   ├── questions.json
│   ├── evaluate.py
│   ├── compare_strategies.py
│   └── results/
├── vectorstore/
│   ├── faiss_fixed/
│   ├── faiss_semantic/
│   └── faiss_hierarchical/
├── output.md                   # 运行日志与实验记录
└── requirements.txt
```

---

## 2. 环境准备

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY="sk-xxx"
```

> 说明：项目使用 DashScope OpenAI 兼容接口进行 embedding / chat 调用。

---

## 3. 运行流程（原生版）

### 3.1 PDF 解析

```bash
python src/parse_pdf.py
```

### 3.2 文档分块

```bash
python src/chunk_documents.py
```

本次记录（`semantic`）：

- 刑法：2165 chunks
- 宪法：1026 chunks
- 民法典：6818 chunks
- 合并后：10009 chunks（`data/chunks/all_semantic.json`）

### 3.3 构建向量索引

```bash
python src/build_index.py
```

输出：

- `vectorstore/faiss_index.bin`（约 40MB）
- `vectorstore/faiss_meta.json`

### 3.4 问答（CLI）

```bash
# 交互式
python src/rag_pipeline.py

# 单次问答
python src/rag_pipeline.py --query "在微信群里骂人犯法吗？侵犯了什么权利？"

# 增加过滤（示例：只看民法典）
python src/rag_pipeline.py --query "小区物业能停水停电催缴物业费吗？" --raw_name "中华人民共和国民法典"

# 开启查询改写
python src/rag_pipeline.py --query "遗嘱" --query-rewrite

# 消融开关
python src/rag_pipeline.py --query "..." --no-bm25
python src/rag_pipeline.py --query "..." --no-rerank
```

---

## 4. 运行流程（LangChain 版）

### 4.1 下载本地 embedding 模型

```bash
python src_langchain/download_model.py
```

### 4.2 构建 LangChain 向量库

```bash
python src_langchain/build_index_lc.py
```

本次记录：

- 加载 571 页（3 个 PDF）
- 分块后 2323 chunks
- 输出路径：`vectorstore/faiss_lc/`（`index.faiss` 约 4.6MB）

### 4.3 LangChain 问答

```bash
# 交互式
python src_langchain/rag_chain_lc.py

# 单次
python src_langchain/rag_chain_lc.py --query "故意伤害致人轻伤会坐牢吗"
```

---

## 5. HTTP 服务

```bash
python src/serve.py
```

启动后访问：

- Web 界面：`http://localhost:8000/`
- API：`POST /query`

Python 调用示例：

```python
import requests

resp = requests.post(
    "http://localhost:8000/query",
    json={"question": "我被前公司领导性骚扰，已经离职了还能告他吗？"},
)
data = resp.json()
print(data["answer"])
for c in data["citations"]:
    print(f"[{c['index']}] {c['source']}")
```

---

## 6. 评估与消融

### 6.1 自动评估

```bash
python evaluation/evaluate.py --pipeline native
python evaluation/evaluate.py --pipeline langchain
```

`native` 题型统计（记录）：

- simple_fact：拒绝率 25%
- precise_number：拒绝率 33%
- cross_doc_compare：拒绝率 40%
- time_trend：拒绝率 67%
- should_refuse：拒绝率 100%

### 6.2 检索消融

```bash
python evaluation/compare_strategies.py
python evaluation/compare_strategies.py --strategies semantic,hierarchical
python evaluation/compare_strategies.py --modes vector_only,hybrid
```

本次记录（Top-4）：

- semantic + vector_only：Hit@4 = 0.700，MRR = 0.700
- semantic + hybrid：Hit@4 = 0.700，MRR = 0.700
- hierarchical + vector_only：Hit@4 = 0.700，MRR = 0.700
- hierarchical + hybrid：Hit@4 = 0.700，MRR = 0.700

结果文件：

- `evaluation/results/ablation_results.json`

---

## 7. 已知现象

- 日志中出现 `faiss.swigfaiss_avx2` 缺失提示通常可忽略，FAISS 会 fallback 到普通实现并继续运行。
- `langchain-community` 有 deprecation warning，不影响当前功能，可后续升级迁移。

---

## 8. 参考

- 详细运行日志：`output.md`
- 依赖清单：`requirements.txt`

## 结果解读（系统是否正常）
本项目当前结果可以判断为**流程正常、系统可用**：
1. **链路完整跑通**：已完成 PDF 解析、分块、向量建库、检索、生成、评估全流程，且各步骤均有可复现输出。
2. **检索有效**：`compare_strategies` 中 Hit@4=0.700、MRR=0.700，说明检索结果能覆盖大部分目标文档，命中机制工作正常。
3. **问答可用**：原生版与 LangChain 版均能返回带来源的答案，拒答逻辑可触发（`should_refuse` 拒答率 100%）。
4. **评估有区分度**：不同题型（simple_fact / precise_number / cross_doc_compare / time_trend）表现存在明显差异，符合真实系统特征。
同时，这些结果也说明系统仍有优化空间（并非“性能封顶”）：
- `cross_doc_compare` 与 `time_trend` 拒答率较高，说明复杂推理与跨文档聚合能力仍待提升。
- 消融实验中不同策略分数接近，后续可扩大题量、增加更细粒度指标（如 nDCG、Recall@k）以拉开差异。