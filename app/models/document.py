"""Internal document model.

Use this if you want a stable internal representation before mapping into
Qdrant payloads.
"""

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """Represents a chunk of source knowledge."""

    id: str
    source: str
    content: str
    metadata: dict[str, str | int | float | bool | None] = {}
