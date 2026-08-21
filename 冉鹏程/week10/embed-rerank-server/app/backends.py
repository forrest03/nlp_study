"""Inference backends."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Protocol, Sequence


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingBatch:
    """Store vectors and token usage returned from a backend."""

    vectors: list[list[float]]
    token_count: int


@dataclass(frozen=True)
class RerankBatch:
    """Store reranker scores and token usage returned from a backend."""

    scores: list[float]
    token_count: int


class EmbeddingBackend(Protocol):
    """Define the backend contract required by the service layer."""

    def warmup(self) -> None:
        """Load external resources required for inference."""

    def encode(self, texts: Sequence[str], normalize: bool) -> EmbeddingBatch:
        """Encode input texts into embeddings."""


class RerankerBackend(Protocol):
    """Define the backend contract required by the reranker service."""

    def warmup(self) -> None:
        """Load external resources required for inference."""

    def score(self, query: str, documents: Sequence[str]) -> RerankBatch:
        """Score query-document pairs."""


class TransformerModelBackend:
    """Share lazy loading and device resolution for local Transformer models."""

    def __init__(self, model_path: str, device: str) -> None:
        """Initialize backend state.

        Args:
            model_path: Local model directory.
            device: Configured inference device name.
        """
        self._model_path = Path(model_path)
        self._device_name = device
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None

    def warmup(self) -> None:
        """Load tokenizer and model before serving traffic."""
        self._load_components()

    def _load_components(self) -> None:
        """Load model assets lazily to keep import-time failures isolated."""
        if self._model is not None and self._tokenizer is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {self._model_path}")
        self._do_load_components()

    def _do_load_components(self) -> None:
        """Implement model-specific loading logic."""
        raise NotImplementedError

    def _resolve_device(self, torch_module):
        """Resolve the configured inference device safely."""
        if self._device_name == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return self._device_name


class HuggingFaceBgeBackend(TransformerModelBackend):
    """Load a local BGE model and generate embeddings with Transformers."""

    def encode(self, texts: Sequence[str], normalize: bool) -> EmbeddingBatch:
        """Encode texts and return vectors with token usage.

        Args:
            texts: Input texts to embed.
            normalize: Whether to L2-normalize the vectors.

        Returns:
            EmbeddingBatch: Result vectors and token usage.
        """
        self._load_components()
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            outputs = self._model(**encoded)
        pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        if normalize:
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors = pooled.cpu().tolist()
        token_count = int(encoded["attention_mask"].sum().item())
        return EmbeddingBatch(vectors=vectors, token_count=token_count)

    def _do_load_components(self) -> None:
        """Load tokenizer and encoder model."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = self._resolve_device(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        self._model = AutoModel.from_pretrained(str(self._model_path))
        self._model.to(self._device)
        self._model.eval()
        LOGGER.info(
            "embedding_model_loaded model=%s device=%s",
            self._model_path,
            self._device,
        )

    def _mean_pool(self, last_hidden_state, attention_mask):
        """Pool token embeddings using the attention mask.

        Why:
            BGE models are commonly pooled with masked mean pooling to
            suppress padding tokens and keep sentence vectors stable.
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class HuggingFaceRerankerBackend(TransformerModelBackend):
    """Load a local BGE reranker model and score query-document pairs."""

    def score(self, query: str, documents: Sequence[str]) -> RerankBatch:
        """Score query-document pairs.

        Args:
            query: Query text.
            documents: Candidate documents.

        Returns:
            RerankBatch: Relevance scores and token usage.
        """
        self._load_components()
        pairs = [[query, document] for document in documents]
        encoded = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            outputs = self._model(**encoded)
        logits = outputs.logits.view(-1).float()
        scores = logits.cpu().tolist()
        token_count = int(encoded["attention_mask"].sum().item())
        return RerankBatch(scores=scores, token_count=token_count)

    def _do_load_components(self) -> None:
        """Load tokenizer and sequence-classification reranker model."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = self._resolve_device(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self._model_path)
        )
        self._model.to(self._device)
        self._model.eval()
        LOGGER.info(
            "reranker_model_loaded model=%s device=%s",
            self._model_path,
            self._device,
        )
