"""Application settings.

Populate `.env` from `.env.example`. Keep all deployment-sensitive values here
so application code reads from one place.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed environment configuration for the service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "MagicRAG"
    app_version: str = "0.1.0"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    rag_documents_dir: Path = Field(default=Path("/data/rag"))
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "magic_knowledge"
    qdrant_vector_size: int = 1024

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"
    deepseek_embedding_model: str = "text-embedding-v3"

    top_k: int = 5
    score_threshold: float = 0.45
    chunk_size: int = 800
    chunk_overlap: int = 120


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
