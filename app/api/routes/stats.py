"""Statistics routes.

Keep this lightweight for now. Later you can aggregate collection counts,
document ingestion status, token usage, and request metrics here.
"""

from fastapi import APIRouter, Depends

from app.schemas.stats import StatsResponse
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter()


@router.get("", response_model=StatsResponse, summary="Service statistics")
async def get_stats(service: RAGService = Depends(get_rag_service)) -> StatsResponse:
    """Return basic service-level statistics."""
    return await service.get_stats()
