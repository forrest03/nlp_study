"""Request and response schemas for the inference API."""

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    """Validate embedding requests from clients."""

    input: Union[str, List[str]] = Field(..., description="Input text or text list.")
    model: Optional[str] = Field(
        default=None,
        description="Optional model name. Defaults to the server-side embedding model.",
    )
    normalize: Optional[bool] = Field(
        default=None,
        description="Whether to L2-normalize the resulting embeddings.",
    )

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: Union[str, List[str]]) -> Union[str, List[str]]:
        """Reject empty text inputs.

        Args:
            value: Raw client input.

        Returns:
            Union[str, List[str]]: The validated input value.

        Raises:
            ValueError: Raised when input is empty or contains blank text.
        """
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("input text must not be blank")
            return value

        if not value:
            raise ValueError("input list must not be empty")

        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("input list contains blank text")

        return value


class EmbeddingData(BaseModel):
    """Represent one embedding item in the response."""

    index: int
    embedding: List[float]


class Usage(BaseModel):
    """Describe token usage for a request."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Return simplified embedding results for internal clients."""

    data: List[EmbeddingData]
    model: str
    usage: Usage


class HealthResponse(BaseModel):
    """Expose server health and loaded model metadata."""

    status: str
    embedding_model: str
    reranker_model: str


class RerankRequest(BaseModel):
    """Validate rerank requests from clients."""

    query: str = Field(..., description="Query text.")
    documents: List[str] = Field(..., description="Candidate documents.")
    model: Optional[str] = Field(
        default=None,
        description="Optional model name. Defaults to the server-side reranker model.",
    )
    return_documents: bool = Field(
        default=True,
        description="Whether to include original documents in the response.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject blank query text."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, value: List[str]) -> List[str]:
        """Reject empty or blank documents."""
        if not value:
            raise ValueError("documents must not be empty")
        for item in value:
            if not item.strip():
                raise ValueError("documents contains blank text")
        return value


class RerankResult(BaseModel):
    """Represent a single reranked document."""

    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    """Return reranker results sorted by relevance."""

    model: str
    results: List[RerankResult]
    usage: Usage
