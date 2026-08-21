"""RAG 系统配置：路径、切分参数、API（生成 + 嵌入）、检索参数。

- 生成：DeepSeek V4-Pro（OpenAI 兼容）
- 嵌入：DashScope text-embedding-v3（OpenAI 兼容）
API Key 从环境变量读取（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY），不写入源码。
"""
import os
from pathlib import Path

# ---- 路径 ----
# 本文件位于 src/ 下，项目根是它的上一级目录
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdf"
MANIFEST_PATH = DATA_DIR / "manifest.json"
INDEX_DIR = ROOT / "index"  # 存放 FAISS 索引与 chunks 元数据

# ---- 切分参数 ----
CHUNK_SIZE = 500       # 每个片段目标字符数
CHUNK_OVERLAP = 80     # 片段间重叠字符数（保留上下文）

# ---- 生成 LLM（DeepSeek，OpenAI 兼容）----
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-v4-pro"
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ---- 嵌入（DashScope，OpenAI 兼容）----
EMBED_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL = "text-embedding-v3"
EMBED_DIM = 1024
EMBED_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
EMBED_BATCH = 10  # 每次请求的文本条数（稳妥上限）

# ---- 检索 ----
# 混合检索：向量检索（语义）+ BM25（关键词）+ RRF 融合排名
TOP_K = 5              # 最终返回给大模型的片段数
CANDIDATE_K = 20       # 每路检索各取的候选数（融合后再裁到 TOP_K）
RRF_K = 60             # RRF 公式常数：score = Σ 1/(RRF_K + rank)，标准值 60
