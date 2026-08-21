"""HTTP routes for the inference server."""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    RerankRequest,
    RerankResponse,
)
from app.service import EmbeddingService, RerankerService


LOGGER = logging.getLogger(__name__)


def build_router(
    embedding_service: EmbeddingService,
    reranker_service: RerankerService,
) -> APIRouter:
    """Build API routes around the application service.

    Args:
        service: Application service handling embedding requests.

    Returns:
        APIRouter: Configured router instance.
    """
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        """Report server readiness and loaded model name.

        Returns:
            HealthResponse: Health payload for monitoring systems.
        """
        return HealthResponse(
            status="ok",
            embedding_model=embedding_service._settings.embedding_model_name,
            reranker_model=reranker_service._settings.reranker_model_name,
        )

    @router.post("/v1/embeddings", response_model=EmbeddingResponse)
    def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        """Handle embedding requests.

        Args:
            request: Validated request body.

        Returns:
            EmbeddingResponse: OpenAI-compatible embedding payload.

        Raises:
            HTTPException: Raised when request or backend processing fails.
        """
        try:
            return embedding_service.create_embeddings(request)
        except ValueError as exc:
            LOGGER.warning("embedding_request_rejected reason=%s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            LOGGER.exception("embedding_model_missing")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("embedding_request_failed")
            raise HTTPException(status_code=500, detail="embedding request failed") from exc

    @router.post("/v1/rerank", response_model=RerankResponse)
    def rerank(request: RerankRequest) -> RerankResponse:
        """Handle reranker requests.

        Args:
            request: Validated request body.

        Returns:
            RerankResponse: Sorted reranker response payload.

        Raises:
            HTTPException: Raised when request or backend processing fails.
        """
        try:
            return reranker_service.rerank(request)
        except ValueError as exc:
            LOGGER.warning("rerank_request_rejected reason=%s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            LOGGER.exception("reranker_model_missing")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("rerank_request_failed")
            raise HTTPException(status_code=500, detail="rerank request failed") from exc

    return router
