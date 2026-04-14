"""Application settings.

Populate `.env` from `.env.example`. Keep all deployment-sensitive values here
so application code reads from one place.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
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

    # Support both the project's DEEPSEEK_* names and OpenAI-style names used
    # by many compatible providers in deployment platforms.
    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"),
    )
    deepseek_chat_model: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("DEEPSEEK_CHAT_MODEL", "OPENAI_MODEL"),
    )
    deepseek_embedding_model: str = Field(
        default="BAAI/bge-large-zh-v1.5",
        validation_alias=AliasChoices(
            "DEEPSEEK_EMBEDDING_MODEL",
            "OPENAI_EMBEDDING_MODEL",
        ),
    )
    embedding_batch_size: int = 32

    top_k: int = 8
    score_threshold: float = 0.45
    chunk_size: int = 800
    chunk_overlap: int = 100
    query_chunk_window: int = 2
    max_context_sources: int = 3
    max_chunks_per_source: int = 6


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
