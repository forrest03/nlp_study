
import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent.parent
print("BASE_DIR:", BASE_DIR)
EVAL_DIR      = Path(__file__).parent
RESULT_DIR    = EVAL_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_MODEL   = "text-embedding-v3"
EMBED_DIM     = 1024
vs_dir = BASE_DIR / "vectorstore" / f"faiss_semantic"
index_path = vs_dir / "index.bin"
meta_path = vs_dir / "meta.json"

def load_index(strategy: str):
    """
    根据策略名加载对应的 FAISS 索引和 meta。
    索引路径约定：
      vectorstore/faiss_{strategy}/index.bin
      vectorstore/faiss_{strategy}/meta.json
    """
    import faiss
    vs_dir     = BASE_DIR / "vectorstore" / f"faiss_{strategy}"
    index_path = vs_dir / "index.bin"
    meta_path  = vs_dir / "meta.json"

    # 兼容默认路径（semantic 策略即主路径）
    if not index_path.exists():
        index_path = BASE_DIR / "vectorstore/faiss_semantic/faiss_index.bin"
        meta_path  = BASE_DIR / "vectorstore/faiss_semantic/faiss_meta.json"
        logger.warning(f"未找到 {vs_dir}，使用默认索引路径（{strategy} 策略）")
        print(index_path)
        print(meta_path)
    print(index_path)
    print(meta_path)
if __name__ == '__main__':
    load_index("semantic")