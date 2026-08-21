# 检索数据库

本目录存放 **BM25 + RAG 向量检索** 使用的结构化数据，由 `memory_compressor` 在定期压缩长期记忆时自动维护。

| 文件 | 说明 |
|------|------|
| `memory_meta.json` | 记忆块元数据（chunk_id、content、tags），供 BM25 分词索引 |
| `memory_faiss_index.bin` | FAISS 向量索引，供语义检索 |

> 人类可读的记忆摘要见 [`memory/compressed/memories.md`](../memory/compressed/memories.md)。
