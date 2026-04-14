from pathlib import Path

import pytest

from app.schemas.rag import ChatTurn, IngestRequest, QueryRequest, SearchRequest
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
async def test_semantic_search_returns_debug_views(monkeypatch: pytest.MonkeyPatch) -> None:
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
                    "source": "/tmp/Darwin Ortiz - Strong Magic.txt",
                    "filename": "Darwin Ortiz - Strong Magic.txt",
                    "chunk_index": 0,
                    "content": "alpha beta gamma",
                },
            )
        ],
    )
    monkeypatch.setattr(
        "app.services.rag_service.fetch_source_chunks",
        lambda source, start, end: [
            _Point(score=0.0, payload={"chunk_index": 0, "content": "alpha beta gamma"})
        ],
    )

    result = await RAGService().semantic_search(SearchRequest(query="Strong Magic", debug=True))

    assert len(result.results) == 1
    assert len(result.selected_results) == 1
    assert len(result.expanded_results) == 1


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


def test_reference_metadata_extracts_numbered_title_and_author() -> None:
    metadata = RAGService._reference_metadata(
        "/tmp/48 Which Hand Fraser Parker.md",
        "48 Which Hand Fraser Parker.md",
    )
    assert metadata["title"] == "Which Hand"
    assert metadata["author"] == "Fraser Parker"


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
        "app.services.rag_service.fetch_source_chunks",
        lambda source, start, end: [
            _Point(
                score=0.0,
                payload={"chunk_index": 0, "content": "alpha beta gamma"},
            ),
            _Point(
                score=0.0,
                payload={"chunk_index": 1, "content": "delta epsilon zeta"},
            ),
        ],
    )
    monkeypatch.setattr(
        RAGService,
        "_semantic_search_with_timings",
        lambda self, payload: _search_response_with_timings(),
    )
    monkeypatch.setattr("app.services.rag_service.complete_chat", _complete_chat)

    result = await RAGService().answer_question(
        QueryRequest(
            question="ignored",
            top_k=3,
            language="en",
            debug=True,
            history=[
                ChatTurn(role="user", content="Tell me about Strong Magic."),
                ChatTurn(role="assistant", content="It is about performance theory."),
            ],
        )
    )

    assert result.used_rag is True
    assert result.fallback_used is False
    assert result.answer == "rag answer"
    assert len(result.sources) == 1
    assert "alpha beta gamma" in result.sources[0].content
    assert "delta epsilon zeta" in result.sources[0].content
    assert result.embedding_time_ms is not None
    assert result.vector_search_time_ms is not None
    assert result.retrieval_time_ms is not None
    assert result.generation_time_ms is not None
    assert result.total_time_ms is not None
    assert len(result.debug_results) == 2
    assert len(result.debug_selected_results) == 2
    assert len(result.debug_expanded_results) == 1
    assert "[Excerpt 1]" in result.debug_expanded_results[0].content


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


def test_expand_results_with_neighbors_pulls_adjacent_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.rag_service.settings.query_chunk_window", 1)
    monkeypatch.setattr("app.services.rag_service.settings.max_context_sources", 2)
    monkeypatch.setattr("app.services.rag_service.settings.max_chunks_per_source", 4)
    monkeypatch.setattr(
        "app.services.rag_service.fetch_source_chunks",
        lambda source, start, end: [
            _Point(score=0.0, payload={"chunk_index": 4, "content": "before"}),
            _Point(score=0.0, payload={"chunk_index": 5, "content": "hit"}),
            _Point(score=0.0, payload={"chunk_index": 6, "content": "after"}),
        ],
    )

    expanded = RAGService._expand_results_with_neighbors(
        [
            _result("Manual of Mystery - Jane Doe.txt", 5, "hit", 0.92),
            _result("Manual of Mystery - Jane Doe.txt", 6, "after", 0.88),
        ]
    )

    assert len(expanded) == 1
    assert expanded[0].metadata["title"] == "Manual of Mystery"
    assert expanded[0].metadata["author"] == "Jane Doe"
    assert "[Excerpt 1] before" in expanded[0].content
    assert "[Excerpt 2] hit" in expanded[0].content
    assert "[Excerpt 3] after" in expanded[0].content


def test_compress_results_for_prompt_keeps_relevant_sentences() -> None:
    compressed = RAGService._compress_results_for_prompt(
        "which hand 的方法",
        [],
        [
            _search_result(
                "48 Which Hand Fraser Parker.md",
                0,
                "[Excerpt 1] This is the opening overview. [Excerpt 2] The method uses three phases. "
                "[Excerpt 3] The spectator hides a coin behind their back. [Excerpt 4] It avoids logic puzzles.",
                0.68,
                title="Which Hand",
                author="Fraser Parker",
            )
        ],
    )

    assert len(compressed) == 1
    assert "three phases" in compressed[0].content
    assert "hides a coin" in compressed[0].content


def test_select_context_results_prefers_named_method_source() -> None:
    selected = RAGService._select_context_results(
        "which hand 的方法",
        [
            _search_result(
                "48 Which Hand Fraser Parker.md",
                0,
                "which hand core method",
                0.68,
                title="Which Hand",
                author="Fraser Parker",
            ),
            _search_result(
                "Corinda’s 13 Steps to Mentalism (1968).txt",
                3037,
                "borrowed paper and cards",
                0.67,
                title="Corinda’s 13 Steps to Mentalism (1968)",
                author=None,
            ),
            _search_result(
                "Darwin Ortiz - Strong Magic.txt",
                2182,
                "timing and shuttle pass",
                0.65,
                title="Strong Magic",
                author="Darwin Ortiz",
            ),
        ],
    )

    assert len(selected) == 1
    assert selected[0].metadata["title"] == "Which Hand"
    assert selected[0].metadata["author"] == "Fraser Parker"


def test_select_context_results_uses_history_for_followup() -> None:
    selected = RAGService._select_context_results(
        "有什么表演技巧吗？",
        [
            _search_result(
                "Darwin Ortiz - Strong Magic.txt",
                64,
                "audience management and showmanship",
                0.51,
                title="Strong Magic",
                author="Darwin Ortiz",
            ),
            _search_result(
                "Corinda’s 13 Steps to Mentalism (1968).txt",
                3037,
                "borrowed paper and cards",
                0.50,
                title="Corinda’s 13 Steps to Mentalism (1968)",
                author=None,
            ),
        ],
        history=[
            ChatTurn(role="user", content="Strong Magic 讲了什么？"),
            ChatTurn(role="assistant", content="它讨论表演理论。"),
        ],
    )

    assert len(selected) == 1
    assert selected[0].metadata["title"] == "Strong Magic"


def test_build_search_query_includes_recent_user_context() -> None:
    query = RAGService._build_search_query(
        "有什么表演技巧吗？",
        [
            ChatTurn(role="user", content="Strong Magic 讲了什么？"),
            ChatTurn(role="assistant", content="它谈的是表演理论。"),
            ChatTurn(role="user", content="我想继续问这本书。"),
        ],
    )
    assert "Strong Magic 讲了什么？" in query
    assert "我想继续问这本书。" in query
    assert "Current question: 有什么表演技巧吗？" in query


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
            ),
            SearchResult(
                source="/tmp/note.md",
                score=0.87,
                content="delta epsilon zeta",
                metadata={
                    "filename": "Darwin Ortiz - Strong Magic.txt",
                    "chunk_index": 1,
                    "title": "Strong Magic",
                    "author": "Darwin Ortiz",
                },
            ),
        ],
    )


async def _empty_search_response():
    from app.schemas.rag import SearchResponse

    return SearchResponse(query="q", results=[])


async def _search_response_with_timings():
    return await _search_response(), 12, 34


async def _empty_search_response_with_timings():
    return await _empty_search_response(), 12, 34


def _result(filename: str, chunk_index: int, content: str, score: float):
    from app.schemas.rag import SearchResult

    return SearchResult(
        source=f"/tmp/{filename}",
        score=score,
        content=content,
        metadata={
            "filename": filename,
            "chunk_index": chunk_index,
            "title": "Manual of Mystery",
            "author": "Jane Doe",
        },
    )


def _search_result(
    filename: str,
    chunk_index: int,
    content: str,
    score: float,
    *,
    title: str,
    author: str | None,
):
    from app.schemas.rag import SearchResult

    return SearchResult(
        source=f"/tmp/{filename}",
        score=score,
        content=content,
        metadata={
            "filename": filename,
            "chunk_index": chunk_index,
            "title": title,
            "author": author,
        },
    )


async def _complete_chat(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    assert "Context:" in user_prompt
    assert "do not use outside knowledge" in system_prompt.lower()
    assert "never introduce a person" in system_prompt.lower()
    assert "according to the file" in system_prompt.lower()
    assert "magic teaching" in system_prompt.lower()
    assert "answer the question completely and in detail" in system_prompt.lower()
    assert "author: Darwin Ortiz" in user_prompt
    assert "title: Strong Magic" in user_prompt
    assert "Recent conversation:" in user_prompt
    assert "User: Tell me about Strong Magic." in user_prompt
    assert "alpha beta gamma" in user_prompt
    assert "delta epsilon zeta" in user_prompt
    assert "Answer guidance:" in user_prompt
    assert "do not guess or substitute one" in system_prompt.lower()
    assert "ui already shows the references" in user_prompt.lower()
    assert "do not add extra recommendations" in user_prompt.lower()
    assert "natural, idiomatic simplified chinese" in user_prompt.lower()
    assert "do not invent or substitute a different author" in user_prompt.lower()
    assert "english" in user_prompt.lower()
    assert max_tokens is not None
    return "rag answer"


def test_term_guide_for_chinese() -> None:
    guide = RAGService._relevant_term_guide(
        "zh",
        "什么是双关语？",
        [],
    )
    assert "Equivoque -> 双关语" in guide
    assert "If one of these terms is needed" in guide


async def _fallback_chat(system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
    assert max_tokens == 360
    return "fallback answer"
