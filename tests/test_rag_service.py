from pathlib import Path

import pytest

from app.schemas.rag import IngestRequest, QueryRequest, SearchRequest
from app.services.rag_service import RAGService


@pytest.mark.asyncio
async def test_ingest_documents_upserts_vectors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_text("alpha beta gamma", encoding="utf-8")

    monkeypatch.setattr("app.services.rag_service.settings.rag_documents_dir", tmp_path)
    monkeypatch.setattr("app.services.rag_service.settings.deepseek_api_key", "key")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_url", "https://example.com")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_api_key", "secret")
    monkeypatch.setattr("app.services.rag_service.settings.embedding_batch_size", 2)

    upserted = []

    monkeypatch.setattr("app.services.rag_service.ensure_collection", lambda force_recreate=False: None)
    monkeypatch.setattr("app.services.rag_service.embed_texts", lambda texts: _embed(texts))
    monkeypatch.setattr("app.services.rag_service.upsert_points", lambda points: upserted.extend(points))
    monkeypatch.setattr("app.services.rag_service.count_points_for_source", lambda source: 0)

    result = await RAGService().ingest_documents(IngestRequest())

    assert result.success is True
    assert result.documents_discovered == 1
    assert result.chunks_indexed == 1
    assert len(upserted) == 1


@pytest.mark.asyncio
async def test_semantic_search_returns_payload_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.rag_service.settings.deepseek_api_key", "key")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_url", "https://example.com")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_api_key", "secret")
    monkeypatch.setattr("app.services.rag_service.embed_texts", lambda texts: _embed(texts))
    monkeypatch.setattr(
        "app.services.rag_service.query_points",
        lambda vector, limit, score_threshold: [
            _Point(
                score=0.91,
                payload={
                    "source": "/tmp/note.md",
                    "filename": "note.md",
                    "chunk_index": 0,
                    "content": "alpha beta gamma",
                },
            )
        ],
    )

    result = await RAGService().semantic_search(SearchRequest(query="alpha"))

    assert len(result.results) == 1
    assert result.results[0].source == "/tmp/note.md"
    assert result.results[0].content == "alpha beta gamma"


@pytest.mark.asyncio
async def test_answer_question_uses_rag_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RAGService,
        "semantic_search",
        lambda self, payload: _search_response(),
    )
    monkeypatch.setattr("app.services.rag_service.complete_chat", _complete_chat)

    result = await RAGService().answer_question(
        QueryRequest(question="ignored", top_k=3)
    )

    assert result.used_rag is True
    assert result.fallback_used is False
    assert result.answer == "rag answer"


@pytest.mark.asyncio
async def test_answer_question_falls_back_without_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RAGService,
        "semantic_search",
        lambda self, payload: _empty_search_response(),
    )
    monkeypatch.setattr("app.services.rag_service.complete_chat", _fallback_chat)

    result = await RAGService().answer_question(
        QueryRequest(question="ignored", top_k=3)
    )

    assert result.used_rag is False
    assert result.fallback_used is True
    assert result.answer == "fallback answer"


async def _embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), 0.0, 1.0] for text in texts]


class _Point:
    def __init__(self, score: float, payload: dict[str, object]) -> None:
        self.score = score
        self.payload = payload


async def _search_response():
    from app.schemas.rag import SearchResponse, SearchResult

    return SearchResponse(
        query="q",
        results=[
            SearchResult(
                source="/tmp/note.md",
                score=0.9,
                content="alpha beta gamma",
                metadata={"filename": "note.md", "chunk_index": 0},
            )
        ],
    )


async def _empty_search_response():
    from app.schemas.rag import SearchResponse

    return SearchResponse(query="q", results=[])


async def _complete_chat(system_prompt: str, user_prompt: str) -> str:
    assert "Context:" in user_prompt
    return "rag answer"


async def _fallback_chat(system_prompt: str, user_prompt: str) -> str:
    return "fallback answer"
