"""RAG-related routes.

These endpoints are intentionally runnable but thin. Each handler delegates to
service classes where you can implement actual ingestion, retrieval, and LLM
orchestration later.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.rag import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, summary="Ingest documents")
async def ingest_documents(
    payload: IngestRequest,
    service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    """Trigger document ingestion from the configured RAG directory."""
    return await service.ingest_documents(payload)


@router.post("/search", response_model=SearchResponse, summary="Semantic search")
async def semantic_search(
    payload: SearchRequest,
    service: RAGService = Depends(get_rag_service),
) -> SearchResponse:
    """Search indexed knowledge chunks by semantic similarity."""
    return await service.semantic_search(payload)


@router.post("/query", response_model=QueryResponse, summary="Answer question")
async def answer_question(
    payload: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer a question with RAG, falling back to DeepSeek general knowledge."""
    try:
        return await service.answer_question(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
