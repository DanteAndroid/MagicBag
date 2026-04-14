"""Statistics response schemas."""

from pydantic import BaseModel


class StatsResponse(BaseModel):
    """Basic service statistics."""

    service: str
    collection_name: str
    documents_dir: str
    indexed_documents: int
    indexed_chunks: int
    embedding_model: str
    chat_model: str
