"""Qdrant client helpers."""

from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create a cached Qdrant client."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )


def ensure_collection(force_recreate: bool = False) -> None:
    """Create the configured collection when needed."""
    client = get_qdrant_client()
    if force_recreate and client.collection_exists(settings.qdrant_collection_name):
        client.delete_collection(settings.qdrant_collection_name)

    if client.collection_exists(settings.qdrant_collection_name):
        return

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(
            size=settings.qdrant_vector_size,
            distance=Distance.COSINE,
        ),
    )


def upsert_points(points: list[PointStruct]) -> None:
    """Upsert a batch of points into the configured collection."""
    if not points:
        return

    get_qdrant_client().upsert(
        collection_name=settings.qdrant_collection_name,
        points=points,
        wait=True,
    )


def count_points() -> int:
    """Return the exact number of points in the configured collection."""
    if not get_qdrant_client().collection_exists(settings.qdrant_collection_name):
        return 0
    return get_qdrant_client().count(
        collection_name=settings.qdrant_collection_name,
        exact=True,
    ).count


def query_points(vector: list[float], limit: int, score_threshold: float) -> list[Any]:
    """Query the configured collection by vector."""
    response = get_qdrant_client().query_points(
        collection_name=settings.qdrant_collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
        score_threshold=score_threshold,
    )
    return list(response.points)
