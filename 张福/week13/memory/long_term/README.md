# 长期记忆

> 跨会话持久化的事实与事件记录。

## 文件说明

| 文件 | 用途 |
|------|------|
| `memories_raw.md` | 原始缓冲区，对话后逐条追加 |
| `../compressed/memories.md` | LLM 压缩后的可读摘要 |

压缩流程：raw 积累 → LLM 分析 → 写入 compressed + 更新 `databases/` 检索索引 → 清空 raw
