"""Request and response models for RAG endpoints."""

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request to ingest files from the configured local RAG directory."""

    force_reindex: bool = Field(
        default=False,
        description="Whether to rebuild the collection even if data already exists.",
    )


class IngestResponse(BaseModel):
    """Response returned after ingestion."""

    success: bool
    message: str
    documents_discovered: int = 0
    chunks_indexed: int = 0


class SearchRequest(BaseModel):
    """Semantic search input."""

    query: str = Field(..., min_length=1, description="Search query.")
    top_k: int | None = Field(default=None, ge=1, le=20)
    debug: bool = False


class SearchResult(BaseModel):
    """Single search hit."""

    source: str
    score: float
    content: str
    metadata: dict[str, str | int | float | bool | None] = {}


class SearchResponse(BaseModel):
    """Search response payload."""

    query: str
    results: list[SearchResult]
    selected_results: list[SearchResult] = []
    expanded_results: list[SearchResult] = []


class ChatTurn(BaseModel):
    """Single turn of lightweight chat history."""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)


class QueryRequest(BaseModel):
    """LLM answer input."""

    question: str = Field(..., min_length=1, description="User question.")
    top_k: int | None = Field(default=None, ge=1, le=20)
    language: str = Field(default="zh", pattern="^(zh|en)$")
    debug: bool = False
    history: list[ChatTurn] = []


class QueryResponse(BaseModel):
    """Answer response payload."""

    question: str
    answer: str
    used_rag: bool
    fallback_used: bool
    sources: list[SearchResult] = []
    embedding_time_ms: int | None = None
    vector_search_time_ms: int | None = None
    retrieval_time_ms: int | None = None
    generation_time_ms: int | None = None
    total_time_ms: int | None = None
    debug_results: list[SearchResult] = []
    debug_selected_results: list[SearchResult] = []
    debug_expanded_results: list[SearchResult] = []
