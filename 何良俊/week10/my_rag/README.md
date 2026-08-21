# README

本指南详述：当你拿到一个**只包含 `data/`、`index/`、`src/` 三个目录**（外加本 `README.md` 与 `requirements.txt`）的本项目时，如何从零搭建运行环境并使用这个 RAG 问答系统。

---

## 1. 项目是什么

一个基于 A 股年报的 RAG（检索增强生成）问答系统。它读取 5 家公司 × 3 年共 15 份年报 PDF，把内容切块、向量化、存入 FAISS 索引；提问时用**混合检索**（向量检索 + BM25 关键词检索 + RRF 融合排名）找最相关的片段，交给大模型生成带来源标注的回答。

- **覆盖公司**：贵州茅台(600519)、五粮液(000858)、中国平安(601318)、宁德时代(300750)、海康威视(002415)
- **年份**：2021、2022、2023
- **生成模型**：DeepSeek `deepseek-v4-pro`（OpenAI 兼容 API）
- **嵌入模型**：DashScope `text-embedding-v3`（1024 维，OpenAI 兼容 API）
- **向量库**：FAISS（本地落盘，无需服务）
- **检索方式**：混合检索 = 稠密向量检索（语义）+ BM25 稀疏检索（关键词）+ RRF 融合排名

---

## 2. 项目结构

```
my_rag/
├── data/              # 原始数据（只读，不要改动）
│   ├── raw_pdf/         # 15 份年报 PDF（数据源）
│   └── manifest.json    # 报告清单：股票代码/公司名/年份/文件名等
├── index/             # 已构建的索引（构建产物，可重建）
│   ├── faiss.index      # FAISS 向量索引（8303 条 × 1024 维）
│   ├── chunks.json      # 文本片段及其元数据（公司/年份/页码）
│   └── bm25_corpus.json # BM25 分词语料（运行时重建 BM25Okapi）
├── src/               # Python 代码（仅 .py 文件）
│   ├── config.py         # 配置：路径、切分参数、API、检索参数
│   ├── build_index.py    # 构建索引：PDF→切分→嵌入→FAISS
│   ├── ask.py            # 交互问答：检索→大模型生成
│   └── test.py           # 批量测试：随机20问，结果存 JSON
├── output/            # 批量测试产物（运行 test.py 生成）
│   └── qa_results.json   # 20 个问题的问答结果（含来源/预览）
├── requirements.txt   # Python 依赖清单
└── README.md          # 本文件
```

> **关键点**：`index/` 已经构建好，正常使用**不需要**重建。只有当 `data/` 里的 PDF 变了、或切分参数改了、或索引损坏时，才需要按第 6 节重建。
> **关于 output/**：运行 `test.py` 会自动创建该目录并写入 `qa_results.json`，再次运行会覆盖。

---

## 3. 前置环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| 操作系统 | WSL2（Ubuntu 22.04）| 推荐。Windows 原生也能跑，但本指南命令按 WSL2 写 |
| Python | 3.10+ | `python3 --version` 检查 |
| 网络 | 能访问 `api.deepseek.com` 和 `dashscope.aliyuncs.com` | 调用 API 用 |
| API Key ① | DeepSeek API Key | 用于生成回答。申请：https://platform.deepseek.com/ |
| API Key ② | DashScope（阿里云百炼）API Key | 用于文本嵌入。申请：https://bailian.console.aliyun.com/ |

> 项目运行依赖两个 key：DeepSeek 负责生成、DashScope 负责嵌入（DeepSeek 官方不提供 embedding 接口）。

---

## 4. 构建/运行步骤（从零开始）

> 以下命令在 **WSL2 终端**里执行。假设项目位于 `/mnt/f/AI/programs/my/my_rag`，请按你的实际路径替换。

### 4.1 进入项目目录

```bash
cd /mnt/f/AI/programs/my/my_rag
```

### 4.2 创建 Python 虚拟环境

在项目根目录创建 `.venv`（与 `data/`、`index/`、`src/` 同级）：

```bash
python3 -m venv .venv
```

这会生成 `.venv/` 目录，把所有依赖装在里面，不污染系统 Python。

### 4.3 安装依赖

```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

依赖只有 4 个：`PyMuPDF`（PDF 解析）、`faiss-cpu`（向量检索）、`numpy`、`requests`。无 torch、无大模型，安装很快。

验证安装成功：

```bash
.venv/bin/python -c "import fitz, faiss, numpy, requests; print('OK')"
```

### 4.4 配置 API Key（写入环境变量）

两个 key 都写入登录 shell 的配置文件 `~/.profile`（**不要**写进 `~/.bashrc`，原因见第 7 节 FAQ）。把下面命令里的 `<你的key>` 替换成真实 key 后执行：

```bash
echo 'export DEEPSEEK_API_KEY="<你的DeepSeek key>"' >> ~/.profile
echo 'export DASHSCOPE_API_KEY="<你的DashScope key>"' >> ~/.profile
source ~/.profile
```

验证两个 key 都已生效（只显示长度，不泄露明文）：

```bash
echo "DEEPSEEK len=${#DEEPSEEK_API_KEY}  DASHSCOPE len=${#DASHSCOPE_API_KEY}"
```

> 安全提示：key 是敏感信息，切勿提交到 git 或写进 `src/` 下的任何代码文件。本项目代码只从环境变量读取，源码里不含 key。

### 4.5（可选）验证 API 可用

在正式运行前，可用 curl 测试两个接口是否通：

```bash
# 测试 DashScope 嵌入
curl -s -X POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings \
  -H "Authorization: Bearer $DASHSCOPE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-v3","input":["测试"]}' | head -c 200

# 测试 DeepSeek 生成
curl -s -X POST https://api.deepseek.com/chat/completions \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"只回复OK"}]}' | head -c 300
```

第一条应返回含 `"embedding":[...]` 的 JSON；第二条应返回含 `"content":"OK"` 的 JSON。

### 4.6 运行问答

由于 `index/` 已经构建好，直接运行 `ask.py` 即可：

```bash
# 方式 A：激活虚拟环境后运行（推荐，交互体验好）
source .venv/bin/activate
cd src
python ask.py
# 用完执行 deactivate 退出虚拟环境

# 方式 B：不激活，直接用 venv 的 python
cd src
../.venv/bin/python ask.py
```

> 注意：`ask.py` 内部有 `import config` 和 `from build_index import ...`，所以**必须在 `src/` 目录下运行**（或设置 `PYTHONPATH=src`），否则会报 `ModuleNotFoundError`。

启动后会看到：

```
[加载] 索引 ...
[就绪] 共 8303 个片段。输入问题开始问答，Ctrl+C 或输入 exit 退出。

问>
```

输入问题，例如：

```
问> 贵州茅台2023年的营业收入是多少？
问> 宁德时代2023年研发投入占营收比例是多少？
问> 中国平安2022年净利润同比变化如何？
```

回答后会列出每条来源片段的公司、年份、页码和相似度。输入 `exit` 或按 `Ctrl+C` 退出。

---

## 5. 配置说明（src/config.py）

如需调整行为，编辑 `src/config.py`：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `CHUNK_SIZE` | 500 | 每个文本片段的目标字符数 |
| `CHUNK_OVERLAP` | 80 | 相邻片段重叠字符数（保留上下文） |
| `LLM_MODEL` | `deepseek-v4-pro` | 生成模型名 |
| `LLM_BASE_URL` | `https://api.deepseek.com` | 生成 API 地址 |
| `EMBED_MODEL` | `text-embedding-v3` | 嵌入模型名 |
| `EMBED_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 嵌入 API 地址 |
| `EMBED_DIM` | 1024 | 嵌入维度（与模型匹配，勿随意改） |
| `EMBED_BATCH` | 10 | 每次请求的文本条数 |
| `TOP_K` | 5 | 最终返回给大模型的片段数 |
| `CANDIDATE_K` | 20 | 混合检索时向量路 / BM25 路各取的候选数（融合后再裁到 TOP_K） |
| `RRF_K` | 60 | RRF 公式常数：score = Σ 1/(RRF_K + rank)，标准值 60 |

> 改了 `CHUNK_SIZE` / `CHUNK_OVERLAP` / `EMBED_MODEL` / `EMBED_DIM` 任一项，**必须重建索引**（见第 7 节），否则维度不匹配会报错。

---

## 6. 检索原理（混合检索 + RRF）

系统采用**混合检索**，互补两类检索的优势，比单一检索召回更全、更准。

### 两路检索
| 路 | 方法 | 擅长 | 短板 |
|---|---|---|---|
| 向量检索（稠密）| DashScope 把 query/片段编码成 1024 维向量，FAISS 内积（=余弦）| 语义相似、同义改写、概括性问题 | 精确数字/代号/专有名词易漏 |
| BM25 检索（稀疏）| jieba 分词后用 BM25Okapi 按词频打分 | 精确关键词匹配（股票代码、金额、ROE 等）| 不懂同义、词面不匹配则漏 |

### RRF 融合排名
两路各自取 `CANDIDATE_K`（默认 20）条候选，按 RRF（Reciprocal Rank Fusion）公式融合：

```
RRF_score(d) = Σ  1 / (RRF_K + rank_i(d))
```

- `rank_i(d)`：文档 d 在第 i 路结果中的名次（从 1 开始）
- `RRF_K`：常数（默认 60），平滑头部与尾部差距，避免某路第 1 名独大
- 同一文档被两路都命中时分数叠加，自然提升
- 融合后按 RRF 分降序，取前 `TOP_K`（默认 5）条喂给大模型

### 为什么用 RRF
- **无需归一化**：向量余弦分（0~1）与 BM25 分（无上限）量纲不同，直接加权需调权重；RRF 只看排名不看绝对分，天然回避这个问题。
- **互补**：语义相关但词面不同的片段靠向量路召回，含精确关键词的片段靠 BM25 路召回，两路都命中的高质量片段被强化。
- **稳健**：单路失效（如 BM25 对纯语义问句召回差）另一路仍能补上，不会整体崩溃。

### 相关参数（src/config.py）
- `TOP_K=5`：最终给大模型的片段数
- `CANDIDATE_K=20`：每路候选数（融合池）
- `RRF_K=60`：RRF 常数

---

## 7. 重建索引（仅在这些情况下需要）

需要重建的场景：`index/` 不存在或损坏、`data/raw_pdf/` 内容变更、修改了切分或嵌入参数。

```bash
source .venv/bin/activate
cd src
python build_index.py
```

流程：读取 `data/manifest.json` → 用 PyMuPDF 从每份 PDF 抽文本 → 切句、贪心合并成片段 → 调 DashScope 嵌入（每批 10 条，带重试）→ 构建 FAISS 索引 → 对全部片段 jieba 分词生成 BM25 语料 → 写入 `index/`（faiss.index、chunks.json、bm25_corpus.json）。

**耗时**：约 10–12 分钟（8303 个片段，主要花在顺序调 DashScope 嵌入；BM25 分词几秒）。期间会打印进度 `[嵌入] N/8303`。

产物会覆盖 `index/` 下的旧文件。重建过程中 `ask.py` 暂时不可用。

---

## 8. 批量测试（test.py）

`src/test.py` 用于自动评测系统的问答能力，无需手动提问。

### 它做什么
1. 从「5 公司 × 3 年 × 24 个财务维度模板」的候选池中，按固定随机种子（seed=42）**抽 20 个不重复问题**，覆盖营收、净利润、研发投入、毛利率、分红、总资产、现金流、每股收益、客户集中度、风险、ROE 等维度。
2. 逐个走完整问答流程（检索 → DeepSeek 生成），复用 `ask.py` 的 `answer_query` 逻辑。
3. 把每个问题的 **问题、回答、来源（含公司/年份/页码/相似度/片段预览）、错误信息** 写入 `output/qa_results.json`。

### 运行

```bash
source .venv/bin/activate
cd src
python test.py
```

### 产物格式（output/qa_results.json）

```jsonc
{
  "generated_at": "2026-07-09 22:31:53",
  "model": "deepseek-v4-pro",
  "embed_model": "text-embedding-v3",
  "retrieval": "hybrid (dense + BM25 + RRF)",
  "top_k": 5,
  "candidate_k": 20,
  "rrf_k": 60,
  "succeeded": 20,   // 成功数
  "failed": 0,       // 失败数（检索/生成出错计入）
  "results": [
    {
      "question": "贵州茅台2023年的营业收入是多少？",
      "answer": "...正文，引用用 [1][2] 序号...",
      "sources": [
        { "idx": 1, "company": "贵州茅台", "year": "2023",
          "page_start": 56, "page_end": 56, "score": 0.0305,
          "preview": "2023 年度，财务报表所示营业收入发生额为…" }
      ],
      "error": ""    // 出错时填错误信息，成功时为空
    }
  ]
}
```

### 说明
- **耗时**：约 5–8 分钟（20 个问题串行调 DeepSeek，思考模式较慢）。
- **随机种子固定**：每次运行生成相同 20 问，便于横向对比。想换题改 `test.py` 里 `generate_questions(20, seed=42)` 的 seed。
- **覆盖**：再次运行会覆盖 `output/qa_results.json`；想保留历史请改名备份。
- **结果判读**：`error` 非空表示该问失败（多为检索未命中或 API 异常）；`answer` 中「无法直接获得」「片段不足」属正常——说明模型诚实拒绝而非编造。

---

## 9. 常见问题（FAQ）

**Q1：运行时报 `错误：未设置 DEEPSEEK_API_KEY 环境变量` / `DASHSCOPE_API_KEY`**
A：环境变量没被当前 shell 读到。先 `source ~/.profile`，再用 `echo $DEEPSEEK_API_KEY` 确认非空。若是新开的 WSL 窗口，登录 shell 会自动读 `~/.profile`；若用 `bash -c`（非登录）则读不到——改用 `bash -lc`。

**Q2：为什么 key 写 `~/.profile` 而不是 `~/.bashrc`？**
A：Ubuntu 的 `~/.bashrc` 开头有一段「非交互 shell 直接 return」，导致 `bash -lc "python ..."` 这类非交互登录 shell 读不到里面的变量。`~/.profile` 对登录 shell 无条件加载，更稳妥。

**Q3：终端里中文显示乱码**
A：在 WSL 终端里直接运行不会乱码。从 Windows PowerShell 经 `wsl` 调用时可能因 GBK 编码乱码——这**只是显示问题**，数据本身是 UTF-8 正确的。加 `PYTHONIOENCODING=utf-8` 可缓解：
```bash
PYTHONIOENCODING=utf-8 ../.venv/bin/python ask.py
```

**Q4：检索结果不太相关 / 跨年对比只命中单年**
A：默认 `TOP_K=5`、`CANDIDATE_K=20`。检索不相关时可在 `config.py` 调大 `CANDIDATE_K`（如 30）给 RRF 更大融合池，或调大 `TOP_K`（如 8）多给大模型些上下文；改后重跑 `ask.py` 即可，无需重建索引。精确关键词类问题靠 BM25 路，语义概括类靠向量路，混合检索已自动互补。

**Q5：`deepseek-v4-pro` 响应里有个 `reasoning_content` 字段**
A：这是该模型的思考链（thinking mode）。代码只取 `message.content` 作为最终回答，思考链不影响输出。若想降低延迟，可在 `config.py` 的 `generate` 请求里加 `"thinking": {"type": "disabled"}`（需模型支持）。

**Q6：调用 API 报 401 / 余额不足**
A：检查 key 是否正确、账户是否有余额。DashScope 嵌入有免费额度；DeepSeek 按量计费。

**Q7：换电脑/重装后怎么恢复**
A：只要保留 `data/`、`index/`、`src/` 三目录，按本指南第 4 节重建 venv、装依赖、配 key 即可。`index/`（含 faiss.index、chunks.json、bm25_corpus.json）可直接复用，不必重新嵌入（省钱省时）。

---

## 10. 文件清单与职责

| 文件 | 职责 | 改动频率 |
|---|---|---|
| `src/config.py` | 集中配置 | 偶尔调参 |
| `src/build_index.py` | 切分 + 嵌入 + BM25 语料 + 建 FAISS 索引 | 极少改 |
| `src/ask.py` | 交互问答主程序（含混合检索+RRF、answer_query 供复用） | 极少改 |
| `src/test.py` | 批量测试：随机20问 → output/qa_results.json | 极少改 |
| `requirements.txt` | 依赖清单 | 加包时改 |
| `README.md` | 本使用指南 | 调整流程时改 |
| `data/raw_pdf/*.pdf` | 原始年报 | 不改 |
| `data/manifest.json` | 报告元数据 | 加新报告时改 |
| `index/faiss.index` | 向量索引 | 重建时覆盖 |
| `index/chunks.json` | 片段文本+元数据 | 重建时覆盖 |
| `index/bm25_corpus.json` | BM25 分词语料 | 重建时覆盖 |
| `output/qa_results.json` | 批量测试结果 | 运行 test.py 覆盖 |
| `.venv/` | 虚拟环境（本指南生成） | 可随时删重建 |

---

## 11. 快速命令速查

```bash
# 一次性环境搭建
cd /mnt/f/AI/programs/my/my_rag
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo 'export DEEPSEEK_API_KEY="<key>"'   >> ~/.profile
echo 'export DASHSCOPE_API_KEY="<key>"'  >> ~/.profile
source ~/.profile

# 日常使用（index 已存在）
source .venv/bin/activate
cd src
python ask.py

# 批量测试（生成 output/qa_results.json）
cd src
python test.py

# 重建索引（按需）
cd src
python build_index.py
```
