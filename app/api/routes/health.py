"""Health check routes.

Add deeper readiness checks here when Qdrant / LLM connectivity needs to be
validated at runtime.
"""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("", summary="Liveness probe")
async def health_check() -> dict[str, str]:
    """Basic liveness endpoint for Railway and external monitoring."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
    }
