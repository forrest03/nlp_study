"""项目路径常量：memory/ 存 Markdown 记忆，databases/ 存 BM25/RAG 检索数据。"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ── 记忆 Markdown 目录 ────────────────────────────────────────────────────────
MEMORY_DIR = BASE_DIR / "memory"
SHORT_TERM_DIR = MEMORY_DIR / "short_term"
LONG_TERM_DIR = MEMORY_DIR / "long_term"
COMPRESSED_MD_DIR = MEMORY_DIR / "compressed"
USER_PROFILE_DIR = MEMORY_DIR / "user_profile"
DAILY_DIR = MEMORY_DIR / "daily"

LONG_TERM_RAW_MD = LONG_TERM_DIR / "memories_raw.md"
COMPRESSED_MD = COMPRESSED_MD_DIR / "memories.md"
USER_PROFILE_MD = USER_PROFILE_DIR / "profile.md"

# ── 检索数据库目录（BM25 + 向量 RAG） ─────────────────────────────────────────
DATABASE_DIR = BASE_DIR / "databases"
MEMORY_META_FILE = DATABASE_DIR / "memory_meta.json"
FAISS_INDEX_FILE = DATABASE_DIR / "memory_faiss_index.bin"
