"""FastAPI application factory."""

from functools import lru_cache

from fastapi import FastAPI

from app.api import build_router
from app.backends import HuggingFaceBgeBackend, HuggingFaceRerankerBackend
from app.logging_config import configure_logging
from app.service import EmbeddingService, RerankerService
from app.settings import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_service() -> EmbeddingService:
    """Build and cache the application service singleton."""
    settings = get_settings()
    backend = HuggingFaceBgeBackend(
        model_path=settings.embedding_model_path,
        device=settings.device,
    )
    return EmbeddingService(backend=backend, settings=settings)


@lru_cache(maxsize=1)
def get_reranker_service() -> RerankerService:
    """Build and cache the reranker service singleton."""
    settings = get_settings()
    backend = HuggingFaceRerankerBackend(
        model_path=settings.reranker_model_path,
        device=settings.device,
    )
    return RerankerService(backend=backend, settings=settings)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    app = FastAPI(title="Embedding Server", version="1.0.0")
    embedding_service = get_service()
    reranker_service = get_reranker_service()
    app.include_router(build_router(embedding_service, reranker_service))

    @app.on_event("startup")
    def warmup_models() -> None:
        """Load both models during startup to fail fast."""
        embedding_service.warmup()
        reranker_service.warmup()

    return app


app = create_app()
