"""Application services for inference requests."""

import logging
from typing import Optional, Sequence

from app.backends import EmbeddingBackend, RerankerBackend
from app.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
)
from app.settings import Settings


LOGGER = logging.getLogger(__name__)


class EmbeddingService:
    """Coordinate request validation rules and backend inference."""

    def __init__(self, backend: EmbeddingBackend, settings: Settings) -> None:
        """Construct the application service.

        Args:
            backend: Inference backend implementation.
            settings: Runtime settings used for validation.
        """
        self._backend = backend
        self._settings = settings

    def warmup(self) -> None:
        """Preload backend resources before requests arrive."""
        self._backend.warmup()

    def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings for the request payload.

        Args:
            request: Validated API request.

        Returns:
            EmbeddingResponse: OpenAI-compatible embedding response.

        Raises:
            ValueError: Raised when the request violates business rules.
        """
        model_name = self._resolve_model_name(request.model)
        texts = self._normalize_inputs(request.input)
        self._validate_batch_size(texts)
        normalize = (
            self._settings.default_normalize_embeddings
            if request.normalize is None
            else request.normalize
        )
        LOGGER.info(
            "embedding_request_received model=%s item_count=%s normalize=%s",
            model_name,
            len(texts),
            normalize,
        )
        batch = self._backend.encode(texts, normalize=normalize)
        response = self._build_response(batch.vectors, batch.token_count)
        LOGGER.info(
            "embedding_request_completed model=%s item_count=%s prompt_tokens=%s",
            model_name,
            len(texts),
            batch.token_count,
        )
        return response

    def _resolve_model_name(self, model_name: Optional[str]) -> str:
        """Resolve the effective embedding model name for the request."""
        resolved_name = model_name or self._settings.embedding_model_name
        if resolved_name != self._settings.embedding_model_name:
            raise ValueError(f"unsupported model: {model_name}")
        return resolved_name

    def _normalize_inputs(self, raw_input) -> list[str]:
        """Convert API input into a uniform text list."""
        if isinstance(raw_input, str):
            return [raw_input]
        return list(raw_input)

    def _validate_batch_size(self, texts: Sequence[str]) -> None:
        """Keep batch sizes bounded to protect memory usage."""
        if len(texts) > self._settings.embedding_max_batch_size:
            raise ValueError(
                "batch size exceeds limit: "
                f"{len(texts)} > {self._settings.embedding_max_batch_size}"
            )

    def _build_response(
        self,
        vectors: list[list[float]],
        token_count: int,
    ) -> EmbeddingResponse:
        """Wrap vectors in an OpenAI-compatible response payload."""
        data = [
            EmbeddingData(index=index, embedding=vector)
            for index, vector in enumerate(vectors)
        ]
        usage = Usage(prompt_tokens=token_count, total_tokens=token_count)
        return EmbeddingResponse(
            data=data,
            model=self._settings.embedding_model_name,
            usage=usage,
        )


class RerankerService:
    """Coordinate request validation rules and reranker inference."""

    def __init__(self, backend: RerankerBackend, settings: Settings) -> None:
        """Construct the application service.

        Args:
            backend: Reranker backend implementation.
            settings: Runtime settings used for validation.
        """
        self._backend = backend
        self._settings = settings

    def warmup(self) -> None:
        """Preload backend resources before requests arrive."""
        self._backend.warmup()

    def rerank(self, request: RerankRequest) -> RerankResponse:
        """Score and sort documents by query relevance.

        Args:
            request: Validated API request.

        Returns:
            RerankResponse: Sorted reranker response.

        Raises:
            ValueError: Raised when the request violates business rules.
        """
        model_name = self._resolve_model_name(request.model)
        self._validate_batch_size(request.documents)
        LOGGER.info(
            "rerank_request_received model=%s item_count=%s return_documents=%s",
            model_name,
            len(request.documents),
            request.return_documents,
        )
        batch = self._backend.score(request.query, request.documents)
        response = self._build_response(request, batch.scores, batch.token_count)
        LOGGER.info(
            "rerank_request_completed model=%s item_count=%s prompt_tokens=%s",
            model_name,
            len(request.documents),
            batch.token_count,
        )
        return response

    def _resolve_model_name(self, model_name: Optional[str]) -> str:
        """Resolve the effective reranker model name for the request."""
        resolved_name = model_name or self._settings.reranker_model_name
        if resolved_name != self._settings.reranker_model_name:
            raise ValueError(f"unsupported model: {model_name}")
        return resolved_name

    def _validate_batch_size(self, documents: Sequence[str]) -> None:
        """Keep reranker request sizes bounded to protect memory usage."""
        if len(documents) > self._settings.reranker_max_batch_size:
            raise ValueError(
                "document count exceeds limit: "
                f"{len(documents)} > {self._settings.reranker_max_batch_size}"
            )

    def _build_response(
        self,
        request: RerankRequest,
        scores: list[float],
        token_count: int,
    ) -> RerankResponse:
        """Wrap reranker scores in a sorted response payload."""
        indexed_scores = self._zip_scores(request.documents, scores, request.return_documents)
        usage = Usage(prompt_tokens=token_count, total_tokens=token_count)
        results = sorted(indexed_scores, key=lambda item: item.relevance_score, reverse=True)
        return RerankResponse(
            model=self._settings.reranker_model_name,
            results=results,
            usage=usage,
        )

    def _zip_scores(
        self,
        documents: Sequence[str],
        scores: Sequence[float],
        return_documents: bool,
    ) -> list[RerankResult]:
        """Combine backend scores with original document indexes."""
        results = []
        for index, score in enumerate(scores):
            document = documents[index] if return_documents else None
            results.append(
                RerankResult(index=index, relevance_score=float(score), document=document)
            )
        return results
