# 记忆工程

本目录存放**人类可读的 Markdown 记忆文件**，按类型分子目录管理。

| 目录 | 文件 | 说明 |
|------|------|------|
| `short_term/` | `session_{id}.md` | 短期记忆：当前会话最近 N 轮对话 |
| `long_term/` | `memories_raw.md` | 长期记忆原始缓冲区（压缩前） |
| `compressed/` | `memories.md` | LLM 压缩后的长期记忆摘要 |
| `user_profile/` | `profile.md` | 用户特征：偏好、习惯、身份 |
| `daily/` | `YYYY-MM-DD.md` | 按自然日记录的事件与对话摘要 |

> BM25 / RAG 检索用的向量索引与元数据存放在项目根目录的 [`databases/`](../databases/) 文件夹。
