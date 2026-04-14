"""Qdrant client helpers."""

from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

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
        client.create_payload_index(
            collection_name=settings.qdrant_collection_name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        client.create_payload_index(
            collection_name=settings.qdrant_collection_name,
            field_name="chunk_index",
            field_schema=PayloadSchemaType.INTEGER,
            wait=True,
        )
        return

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(
            size=settings.qdrant_vector_size,
            distance=Distance.COSINE,
        ),
    )
    client.create_payload_index(
        collection_name=settings.qdrant_collection_name,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )
    client.create_payload_index(
        collection_name=settings.qdrant_collection_name,
        field_name="chunk_index",
        field_schema=PayloadSchemaType.INTEGER,
        wait=True,
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


def count_points_for_source(source: str) -> int:
    """Return how many points already exist for a given source file."""
    if not get_qdrant_client().collection_exists(settings.qdrant_collection_name):
        return 0

    return get_qdrant_client().count(
        collection_name=settings.qdrant_collection_name,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source),
                )
            ]
        ),
        exact=True,
    ).count


def fetch_source_chunks(source: str, start_chunk: int, end_chunk: int) -> list[Any]:
    """Fetch a contiguous chunk window for one source file."""
    if not get_qdrant_client().collection_exists(settings.qdrant_collection_name):
        return []

    client = get_qdrant_client()
    limit = max(1, end_chunk - start_chunk + 1)
    filtered_records: list[Any]

    try:
        filtered_records, _ = client.scroll(
            collection_name=settings.qdrant_collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source),
                    ),
                    FieldCondition(
                        key="chunk_index",
                        range=Range(gte=start_chunk, lte=end_chunk),
                    ),
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except UnexpectedResponse as exc:
        if "Index required but not found" not in str(exc):
            raise

        # Older collections may not have a chunk_index payload index yet.
        # Fall back to scrolling by source and filtering chunk_index in Python.
        filtered_records = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=settings.qdrant_collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source),
                        )
                    ]
                ),
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                chunk_index = int(record.payload.get("chunk_index", -1))
                if start_chunk <= chunk_index <= end_chunk:
                    filtered_records.append(record)
            if offset is None:
                break

    return sorted(
        filtered_records,
        key=lambda record: int(record.payload.get("chunk_index", 0)),
    )
