# week10检索增强生成/qa_system/config.py
"""集中配置：路径、模型、检索/生成参数。"""
import os
from pathlib import Path

# qa_system/config.py -> parents[0]=qa_system, parents[1]=week10检索增强生成, parents[2]=仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]
QA_DIR = Path(__file__).resolve().parent
INDEX_DIR = QA_DIR / "index"

# 知识库：10 个 weekN.md + 名词解释.md（相对仓库根）
KB_FILES = [
    REPO_ROOT / "week1基本介绍" / "week1.md",
    REPO_ROOT / "week2深度学习基础" / "week2.md",
    REPO_ROOT / "week3深度学习组件" / "week3 深度学习常用组件" / "week3.md",
    REPO_ROOT / "week4语言模型" / "week4 语言模型" / "week4.md",
    REPO_ROOT / "week5大语言模型初探" / "week5.md",
    REPO_ROOT / "week6文本分类问题" / "week6 文本分类问题" / "week6.md",
    REPO_ROOT / "week7 序列标注问题" / "序列标注项目" / "week7.md",
    REPO_ROOT / "week8文本匹配问题" / "week8 文本匹配问题" / "文本匹配项目" / "week8.md",
    REPO_ROOT / "week9大模型应用补充知识" / "week9.md",
    REPO_ROOT / "week10检索增强生成" / "week10 检索增强生成RAG" / "week10.md",
    REPO_ROOT / "名词解释.md",
]

# 硅基流动 SiliconFlow（OpenAI 兼容）
API_BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY_ENV = "SILICONFLOW_API_KEY"

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024               # bge-m3 原生维度（固定，不通过 dimensions 参数指定）
EMBED_BATCH_SIZE = 10          # 每批条数（保守值，兼容各家上限）

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LLM_TEMPERATURE = 0.1

# 检索/融合/阈值
CHUNK_MAX_SIZE = 800
CHUNK_OVERLAP = 64
TOP_K_RECALL = 10              # 各路召回数
TOP_K_FINAL = 5               # 送 LLM 的块数
RRF_K = 60
SCORE_THRESHOLD = 0.25        # 用原始 vec_score（余弦）判断是否拒答


def get_api_key() -> str:
    key = os.getenv(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"环境变量 {API_KEY_ENV} 未设置。请先设置硅基流动 API Key：\n"
            f"  Windows(bash): export {API_KEY_ENV}=sk-xxxx"
        )
    return key
