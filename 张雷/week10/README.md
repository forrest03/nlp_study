# Week10 — RAG 检索增强生成

年报/论文 PDF 的 RAG 问答系统，包含文档解析、分块、向量索引和 LLM 生成。

提供两种实现方式：
- **原生版**（`src/`）：手写流程控制，透明可调试
- **LangChain 版**（`src_langchain/`）：LCEL 链式编排，代码更简洁

## 项目结构

```
week10/
├── src/                       # 原生版 RAG
│   ├── download_reports.py   # 下载年报 PDF（从巨潮资讯网）
│   ├── parse_pdf.py           # PDF → 结构化 JSON（表格+文字+章节）
│   ├── chunk_documents.py     # 语义分块（fixed/semantic/hierarchical 三种策略）
│   ├── build_index.py         # DashScope embedding + FAISS / ChromaDB 索引
│   ├── rag_pipeline.py        # RAG 问答流水线（向量+BM25+RRF+Rerank+LLM）
│   ├── serve.py               # FastAPI HTTP 服务
│   └── static/
│       └── index.html         # 教学可视化页面
├── src_langchain/              # LangChain 版 RAG
│   ├── download_model.py      # 下载本地 BGE embedding 模型
│   ├── build_index_lc.py      # LangChain 全链路建库（Loader → Splitter → FAISS）
│   └── rag_chain_lc.py        # LCEL 链式问答
├── models/
│   └── bge-small-zh-v1.5/     # 本地 BGE embedding 模型（~90MB）
├── data/
│   ├── raw_pdf/               # 原始 PDF 文件
│   ├── parsed/                # 解析后的 JSON
│   ├── chunks/                # 分块 JSON
│   └── manifest.json          # 下载清单（可选）
├── vectorstore/
│   ├── faiss_index.bin        # 原生版 FAISS 索引
│   ├── faiss_meta.json        # 原生版向量元数据
│   └── faiss_lc/              # LangChain 版 FAISS 索引
│       └── index.faiss
├── requirements.txt           # 依赖
└── README.md
```

## 环境准备

```bash
# 1. 创建 conda 环境
conda activate py312

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 DashScope API Key
export DASHSCOPE_API_KEY="sk-xxx"
```

### 系统级依赖（可选）

```bash
# OCR 扫描件需要 tesseract
brew install tesseract
```

## 流水线

### 步骤 1：下载年报 PDF（可选，已有数据则跳过）

```bash
python src/download_reports.py
```

从巨潮资讯网下载 5 家公司（贵州茅台、五粮液、中国平安、宁德时代、海康威视）2021-2023 年年报 PDF 到 `data/raw_pdf/`。

### 步骤 2：解析 PDF

```bash
python src/parse_pdf.py
```

- 用 **pdfplumber** 提取表格（转 Markdown）
- 用 **PyMuPDF** 提取带字体信息的文字（判断标题层级）
- 对扫描页降级为 OCR（pytesseract）
- 输出结构化 JSON 到 `data/parsed/`，保留页码、章节路径等元信息

### 步骤 3：文档分块

```bash
python src/chunk_documents.py
```

三种策略可选（默认 `semantic`）：

| 策略 | 说明 |
|------|------|
| `fixed` | 固定大小 500 字符，重叠 50 |
| `semantic` | 按标题/表格边界智能切分，保留语义完整 |
| `hierarchical` | 父子块结构，子块召回+父块供给 LLM 上下文 |

输出到 `data/chunks/`，合并为 `all_semantic.json`。

### 步骤 4：构建向量索引

```bash
python src/build_index.py
```

- **Embedding**：阿里云 DashScope `text-embedding-v3`，维度 1024
- **向量库**：FAISS `IndexFlatIP`（内积=余弦相似度）
- 可选 ChromaDB 对比（支持元数据过滤）
- 输出 `vectorstore/faiss_index.bin` + `vectorstore/faiss_meta.json`

### 步骤 5：启动服务

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

或交互式命令行：

```bash
python src/rag_pipeline.py              # 交互式问答
python src/rag_pipeline.py --query "xxx" # 单次查询
```

---

## LangChain 版（src_langchain/）

使用 **本地 BGE 模型**做 embedding，**LangChain 全家桶**统一管理 Loader、Splitter、FAISS 和 LCEL 链路。

### 步骤 1：下载 BGE 模型（可选，已有则跳过）

```bash
python src_langchain/download_model.py
```

约 90MB，下载到 `models/bge-small-zh-v1.5/`。可从外部硬盘复制已有模型跳过此步。

### 步骤 2：构建 LangChain 向量库

```bash
python src_langchain/build_index_lc.py
```

- **Loader**：`PyMuPDFLoader` 逐页加载 PDF，每页一个 Document
- **Splitter**：`RecursiveCharacterTextSplitter`，chunk_size=500，overlap=50
- **Embedding**：`HuggingFaceEmbeddings` + 本地 `BAAI/bge-small-zh-v1.5`（维度 512）
- **向量库**：`FAISS.from_documents()` 一行建库
- 输出 `vectorstore/faiss_lc/`

### 步骤 3：运行 LCEL 问答

```bash
# 交互式
python src_langchain/rag_chain_lc.py

# 单次查询
python src_langchain/rag_chain_lc.py --query "这篇论文研究了什么内容"

# 附带来源文档
python src_langchain/rag_chain_lc.py --query "这篇论文用了什么方法" --with-sources
```

### LCEL 数据流

```
question
   │
   ├── retriever → format_docs → "context"
   └── RunnablePassthrough() → "question"
   │
   └──────────┬──────────┘
              │
         {"context": ..., "question": ...}
              │
           prompt
              │
             llm
              │
       StrOutputParser
              │
          answer (str)
```

### 与原生版对比

| 环节 | 原生版 (`src/`) | LangChain 版 (`src_langchain/`) |
|------|-----------------|--------------------------------|
| Embedding | DashScope API `text-embedding-v3` | 本地 `bge-small-zh-v1.5` |
| 维度 | 1024 | 512 |
| 分块 | 语义分块（感知标题/表格） | RecursiveCharacterTextSplitter |
| 检索 | FAISS + BM25 + RRF 混合 | FAISS 单路 |
| Rerank | CrossEncoder 可选 | 无 |
| 链组织 | 手写流程控制 | LCEL `\|` 操作符 |
| 代码量 | ~300 行 | ~120 行 |
| 外部依赖 | 需要 DASHSCOPE_API_KEY | 无需 API Key（本地 embedding） |

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 可视化 Web 页面 |
| GET | `/health` | 健康检查 |
| POST | `/query` | 标准问答，返回答案+引用 |
| POST | `/query/debug` | 调试接口，逐步展示检索中间结果 |

### 请求示例

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "这篇文章研究了什么内容"}'
```

## RAG 流水线技术栈

```
查询改写（qwen-turbo，可选）
       ↓
向量检索（DashScope embedding + FAISS）
       +
BM25 关键词检索（jieba + rank_bm25）
       ↓
RRF 融合排名（k=60）
       ↓
CrossEncoder Rerank（BAAI/bge-reranker-base，可选）
       ↓
相关性阈值过滤（<0.25 拒绝回答）
       ↓
LLM 生成（DashScope qwen-plus）+ 引用标注
```
