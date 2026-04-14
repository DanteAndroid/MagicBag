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
async def test_semantic_search_extracts_reference_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.rag_service.settings.deepseek_api_key", "key")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_url", "https://example.com")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_api_key", "secret")
    monkeypatch.setattr("app.services.rag_service.embed_texts", lambda texts: _embed(texts))
    monkeypatch.setattr(
        "app.services.rag_service.query_points",
        lambda vector, limit, score_threshold: [
            _Point(
                score=0.88,
                payload={
                    "source": "/tmp/Darwin Ortiz - Strong Magic.txt",
                    "filename": "Darwin Ortiz - Strong Magic.txt",
                    "chunk_index": 1,
                    "content": "strong magic excerpt",
                },
            )
        ],
    )

    result = await RAGService().semantic_search(SearchRequest(query="strong magic"))

    assert result.results[0].metadata["title"] == "Strong Magic"
    assert result.results[0].metadata["author"] == "Darwin Ortiz"


@pytest.mark.asyncio
async def test_semantic_search_expands_glossary_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def _capture_embed(texts: list[str]) -> list[list[float]]:
        captured["query"] = texts[0]
        return [[1.0, 0.0, 1.0]]

    monkeypatch.setattr("app.services.rag_service.settings.deepseek_api_key", "key")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_url", "https://example.com")
    monkeypatch.setattr("app.services.rag_service.settings.qdrant_api_key", "secret")
    monkeypatch.setattr("app.services.rag_service.embed_texts", _capture_embed)
    monkeypatch.setattr("app.services.rag_service.query_points", lambda vector, limit, score_threshold: [])

    await RAGService().semantic_search(SearchRequest(query="什么是双关语？"))

    assert "Equivoque" in captured["query"]


@pytest.mark.asyncio
async def test_answer_question_uses_rag_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RAGService,
        "_semantic_search_with_timings",
        lambda self, payload: _search_response_with_timings(),
    )
    monkeypatch.setattr("app.services.rag_service.complete_chat", _complete_chat)

    result = await RAGService().answer_question(
        QueryRequest(question="ignored", top_k=3, language="en")
    )

    assert result.used_rag is True
    assert result.fallback_used is False
    assert result.answer == "rag answer"
    assert result.embedding_time_ms is not None
    assert result.vector_search_time_ms is not None
    assert result.retrieval_time_ms is not None
    assert result.generation_time_ms is not None
    assert result.total_time_ms is not None


@pytest.mark.asyncio
async def test_answer_question_falls_back_without_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RAGService,
        "_semantic_search_with_timings",
        lambda self, payload: _empty_search_response_with_timings(),
    )
    monkeypatch.setattr("app.services.rag_service.complete_chat", _fallback_chat)

    result = await RAGService().answer_question(
        QueryRequest(question="ignored", top_k=3, language="zh")
    )

    assert result.used_rag is False
    assert result.fallback_used is True
    assert result.answer == "fallback answer"
    assert result.embedding_time_ms is not None
    assert result.vector_search_time_ms is not None
    assert result.retrieval_time_ms is not None
    assert result.generation_time_ms is not None
    assert result.total_time_ms is not None


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
                metadata={
                    "filename": "Darwin Ortiz - Strong Magic.txt",
                    "chunk_index": 0,
                    "title": "Strong Magic",
                    "author": "Darwin Ortiz",
                },
            )
        ],
    )


async def _empty_search_response():
    from app.schemas.rag import SearchResponse

    return SearchResponse(query="q", results=[])


async def _search_response_with_timings():
    return await _search_response(), 12, 34


async def _empty_search_response_with_timings():
    return await _empty_search_response(), 12, 34


async def _complete_chat(system_prompt: str, user_prompt: str) -> str:
    assert "Context:" in user_prompt
    assert "do not use outside knowledge" in system_prompt.lower()
    assert "never introduce a person" in system_prompt.lower()
    assert "author: Darwin Ortiz" in user_prompt
    assert "title: Strong Magic" in user_prompt
    assert "do not guess or substitute one" in system_prompt.lower()
    assert "do not add extra recommendations" in user_prompt.lower()
    assert "natural, idiomatic simplified chinese" in user_prompt.lower()
    assert "do not invent or substitute a different author" in user_prompt.lower()
    assert "english" in user_prompt.lower()
    return "rag answer"


def test_term_guide_for_chinese() -> None:
    guide = RAGService._relevant_term_guide(
        "zh",
        "什么是双关语？",
        [],
    )
    assert "Equivoque -> 双关语" in guide
    assert "If one of these terms is needed" in guide


async def _fallback_chat(system_prompt: str, user_prompt: str) -> str:
    return "fallback answer"
