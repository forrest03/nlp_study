"""Unit tests for the inference services."""

from app.backends import EmbeddingBatch, RerankBatch
from app.schemas import EmbeddingRequest, RerankRequest
from app.service import EmbeddingService, RerankerService
from app.settings import Settings


class FakeEmbeddingBackend:
    """Provide deterministic vectors for embedding service tests."""

    def __init__(self) -> None:
        self.warmup_called = False
        self.calls = []

    def warmup(self) -> None:
        """Record warmup invocation."""
        self.warmup_called = True

    def encode(self, texts, normalize):
        """Return deterministic vectors for assertions."""
        self.calls.append((list(texts), normalize))
        vectors = [[float(index), float(index) + 0.5] for index, _ in enumerate(texts)]
        return EmbeddingBatch(vectors=vectors, token_count=7)


class FakeRerankerBackend:
    """Provide deterministic scores for reranker service tests."""

    def __init__(self) -> None:
        self.warmup_called = False
        self.calls = []

    def warmup(self) -> None:
        """Record warmup invocation."""
        self.warmup_called = True

    def score(self, query, documents):
        """Return deterministic scores for assertions."""
        self.calls.append((query, list(documents)))
        return RerankBatch(scores=[0.2, 0.9, 0.5][: len(documents)], token_count=11)


def build_settings() -> Settings:
    """Create test settings with stable limits."""
    return Settings(
        embedding_model_name="bge-large-zh-v1.5",
        embedding_model_path="/tmp/embedding-model",
        reranker_model_name="bge-reranker-v2-m3",
        reranker_model_path="/tmp/reranker-model",
        host="0.0.0.0",
        port=8000,
        device="cpu",
        embedding_max_batch_size=2,
        reranker_max_batch_size=3,
        default_normalize_embeddings=True,
    )


def test_create_embeddings_single_input_uses_default_normalization():
    service = EmbeddingService(backend=FakeEmbeddingBackend(), settings=build_settings())
    request = EmbeddingRequest(input="测试文本")

    response = service.create_embeddings(request)

    assert response.model == "bge-large-zh-v1.5"
    assert response.usage.prompt_tokens == 7
    assert response.data[0].embedding == [0.0, 0.5]


def test_create_embeddings_list_input_respects_override():
    backend = FakeEmbeddingBackend()
    service = EmbeddingService(backend=backend, settings=build_settings())
    request = EmbeddingRequest(
        input=["甲", "乙"],
        model="bge-large-zh-v1.5",
        normalize=False,
    )

    response = service.create_embeddings(request)

    assert len(response.data) == 2
    assert backend.calls == [(["甲", "乙"], False)]


def test_create_embeddings_rejects_oversized_batch():
    service = EmbeddingService(backend=FakeEmbeddingBackend(), settings=build_settings())
    request = EmbeddingRequest(
        input=["甲", "乙", "丙"],
        model="bge-large-zh-v1.5",
    )

    try:
        service.create_embeddings(request)
    except ValueError as exc:
        assert "batch size exceeds limit" in str(exc)
        return

    assert False, "expected ValueError"


def test_rerank_sorts_scores_descending():
    backend = FakeRerankerBackend()
    service = RerankerService(backend=backend, settings=build_settings())
    request = RerankRequest(
        query="招标文件要求",
        documents=["文档A", "文档B", "文档C"],
    )

    response = service.rerank(request)

    assert [item.index for item in response.results] == [1, 2, 0]
    assert response.results[0].document == "文档B"
    assert backend.calls == [("招标文件要求", ["文档A", "文档B", "文档C"])]


def test_rerank_can_hide_documents():
    service = RerankerService(backend=FakeRerankerBackend(), settings=build_settings())
    request = RerankRequest(
        query="评分标准",
        documents=["甲", "乙"],
        return_documents=False,
    )

    response = service.rerank(request)

    assert all(item.document is None for item in response.results)


def test_rerank_rejects_oversized_document_list():
    service = RerankerService(backend=FakeRerankerBackend(), settings=build_settings())
    request = RerankRequest(
        query="评分标准",
        documents=["甲", "乙", "丙", "丁"],
        model="bge-reranker-v2-m3",
    )

    try:
        service.rerank(request)
    except ValueError as exc:
        assert "document count exceeds limit" in str(exc)
        return

    assert False, "expected ValueError"


def test_create_embeddings_rejects_unsupported_model():
    service = EmbeddingService(backend=FakeEmbeddingBackend(), settings=build_settings())
    request = EmbeddingRequest(input="测试文本", model="other-model")

    try:
        service.create_embeddings(request)
    except ValueError as exc:
        assert str(exc) == "unsupported model: other-model"
        return

    assert False, "expected ValueError"


def test_rerank_rejects_unsupported_model():
    service = RerankerService(backend=FakeRerankerBackend(), settings=build_settings())
    request = RerankRequest(
        query="评分标准",
        documents=["甲", "乙"],
        model="other-model",
    )

    try:
        service.rerank(request)
    except ValueError as exc:
        assert str(exc) == "unsupported model: other-model"
        return

    assert False, "expected ValueError"
