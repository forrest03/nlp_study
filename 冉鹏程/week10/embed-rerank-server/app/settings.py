"""Runtime settings for the embedding server."""

from dataclasses import dataclass
import os


DEFAULT_EMBEDDING_MODEL_PATH = "/data/modelscope_cache/models/BAAI/bge-large-zh-v1.5"
DEFAULT_RERANKER_MODEL_PATH = "/data/modelscope_cache/models/BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class Settings:
    """Store immutable runtime settings for the service."""

    embedding_model_name: str
    embedding_model_path: str
    reranker_model_name: str
    reranker_model_path: str
    host: str
    port: int
    device: str
    embedding_max_batch_size: int
    reranker_max_batch_size: int
    default_normalize_embeddings: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables.

        Returns:
            Settings: Parsed immutable settings.
        """
        return cls(
            embedding_model_name="bge-large-zh-v1.5",
            embedding_model_path=os.getenv(
                "EMBEDDING_MODEL_PATH",
                DEFAULT_EMBEDDING_MODEL_PATH,
            ),
            reranker_model_name="bge-reranker-v2-m3",
            reranker_model_path=os.getenv(
                "RERANKER_MODEL_PATH",
                DEFAULT_RERANKER_MODEL_PATH,
            ),
            host=os.getenv("EMBEDDING_HOST", "0.0.0.0"),
            port=int(os.getenv("EMBEDDING_PORT", "8000")),
            device=os.getenv("EMBEDDING_DEVICE", "auto"),
            embedding_max_batch_size=int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "32")),
            reranker_max_batch_size=int(os.getenv("RERANKER_MAX_BATCH_SIZE", "64")),
            default_normalize_embeddings=(
                os.getenv("EMBEDDING_NORMALIZE", "true").lower() != "false"
            ),
        )
