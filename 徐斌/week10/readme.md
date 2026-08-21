# 审核专家 API 文档 RAG 问答系统

基于《审核专家 API 接入文档 V1.0.8》构建的知识库问答系统，支持命令行交互与 Web 可视化界面。

## 架构

```
PDF 解析 → 语义分块 → DashScope Embedding → FAISS 向量索引
                                              ↓
用户提问 → 向量检索 + BM25 → RRF 融合 → qwen-plus 生成答案
```

## 环境准备

```bash
cd 徐斌/week10
pip install -r requirements.txt
export DASHSCOPE_API_KEY="sk-xxx"   # 阿里云 DashScope API Key
```

## 构建知识库（首次或文档更新后执行）

```bash
cd src

# 1. 解析 PDF → 结构化 JSON
python3.11 parse_pdf.py

# 2. 语义分块
python3.11 chunk_documents.py

# 3. 向量化并构建 FAISS 索引
python3.11 build_index.py
```

文档清单配置在 `data/manifest.json`，当前已包含：

- `审核专家 API 接入文档  V1.0.8.pdf`

## 使用方式

### 命令行问答

```bash
cd src
python3.11 rag_pipeline.py

# 单次提问
python3.11 rag_pipeline.py --query "审核专家 API 的鉴权方式是什么？"
python3.11 rag_pipeline.py --query "错误码 10001 代表什么？" --section "错误码"
```

### Web 服务

```bash
cd src
uvicorn serve:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000 即可使用可视化问答界面，可查看向量检索、BM25、RRF 融合、LLM 生成各步骤的中间结果。

API 接口：

| 路径 | 说明 |
|------|------|
| `POST /query` | 标准问答，返回答案 + 引用 |
| `POST /query/debug` | 调试接口，返回各检索阶段中间结果 |
| `GET /health` | 健康检查 |

请求示例：

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "如何调用审核接口？"}'
```

## 目录结构

```
week10/
├── data/
│   ├── manifest.json          # 文档清单
│   ├── 审核专家 API 接入文档  V1.0.8.pdf
│   ├── parsed/                # PDF 解析结果
│   └── chunks/                # 分块结果
├── vectorstore/
│   ├── faiss_index.bin        # FAISS 向量索引
│   └── faiss_meta.json        # chunk 元数据
├── src/
│   ├── parse_pdf.py           # PDF 解析
│   ├── chunk_documents.py     # 文档分块
│   ├── build_index.py         # 向量索引构建
│   ├── rag_pipeline.py        # RAG 问答流水线
│   ├── serve.py               # FastAPI 服务
│   └── static/index.html      # Web 前端
└── requirements.txt
```

## 代码块处理

PDF 中的代码示例会经过专门识别与独立分块：

1. **识别**（`parse_pdf.py`）
   - 等宽字体检测（如 `SourceCodePro-Regular`）
   - 「代码块」标记 + 语言标签（Go / Python / Java …）
   - JSON 返回示例自动识别为 `json`
   - 输出 `block_type: "code"`，内容包在 ` ```语言 ` 围栏中

2. **分块**（`chunk_documents.py`）
   - 每个代码块**独立成 chunk**，不与说明文字合并，也不在中间截断
   - 元数据保留 `code_lang` 字段

3. **检索**
   - 代码块与 prose 一样走向量 + BM25 混合检索
   - 引用来源会标注语言，如「python 代码 · 第3页」

当前知识库：**50 个独立代码块**（Go / Python / Java / JSON 等）

## 当前知识库规模

- 解析块：266 个（含 50 个代码块）
- 语义分块：128 个 chunk（52 表格 + 50 代码 + 26 文本）
- 向量维度：1024（text-embedding-v3）
