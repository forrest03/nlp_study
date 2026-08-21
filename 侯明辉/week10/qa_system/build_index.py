# week10检索增强生成/qa_system/build_index.py
"""建库：遍历 KB_FILES → 分块 → embedding → 存 index/。"""
import json
import sys

import numpy as np

import config
import chunker
import embedder

# Windows 控制台常为 GBK，遇到非 GBK 字符（如 LLM 输出的特殊符号）打印会崩溃。
# 保留控制台原生编码（中文仍正常），仅把无法编码的字符替换为占位符，避免崩溃。
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass


def build() -> int:
    all_chunks = []
    for path in config.KB_FILES:
        if not path.exists():
            print(f"[跳过] 不存在的文件：{path}")
            continue
        cs = chunker.chunk_markdown(path)
        all_chunks.extend(cs)
        print(f"  {path.name}: {len(cs)} 块")

    if not all_chunks:
        raise RuntimeError("未产生任何块，请检查 KB_FILES 路径。")

    print(f"共 {len(all_chunks)} 块，开始向量化 ...")
    embeddings = embedder.embed_texts([c["content"] for c in all_chunks])

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", embeddings)
    (config.INDEX_DIR / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[完成] 索引已保存到 {config.INDEX_DIR}（{embeddings.shape[0]} × {embeddings.shape[1]}）")
    return len(all_chunks)


if __name__ == "__main__":
    build()
