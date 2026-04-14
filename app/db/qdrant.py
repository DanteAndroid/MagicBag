"""Qdrant client helpers.

This module centralizes Qdrant Cloud connectivity. When you implement real
collection creation and upsert logic, keep it here instead of scattering raw
client calls across route handlers.
"""

from functools import lru_cache

from qdrant_client import QdrantClient

from app.core.config import settings


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create a cached Qdrant client.

    TODO: Add collection bootstrap logic here if you want startup-time checks.
    """
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
