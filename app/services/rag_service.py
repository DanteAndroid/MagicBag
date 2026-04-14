"""RAG service orchestration.

This is the main place to implement ingestion, retrieval, reranking, prompt
assembly, and LLM fallback behavior.
"""

from functools import lru_cache
from uuid import uuid5, NAMESPACE_URL

from qdrant_client.models import PointStruct

from app.core.config import settings
from app.db.qdrant import (
    count_points,
    count_points_for_source,
    ensure_collection,
    query_points,
    upsert_points,
)
from app.llm.deepseek import complete_chat, embed_texts
from app.schemas.rag import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.schemas.stats import StatsResponse
from app.services.document_loader import chunk_document, discover_documents


class RAGService:
    """Service object for RAG workflows."""

    @staticmethod
    def _require_ingest_dependencies() -> str | None:
        if not settings.deepseek_api_key:
            return "DEEPSEEK_API_KEY is not configured."
        if not settings.qdrant_url:
            return "QDRANT_URL is not configured."
        if not settings.qdrant_api_key:
            return "QDRANT_API_KEY is not configured."
        return None

    @staticmethod
    def _build_point_id(source: str, chunk_index: int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{source}#{chunk_index}"))

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        parts: list[str] = []
        for index, result in enumerate(results, start=1):
            parts.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"file: {result.metadata.get('filename') or result.source}",
                        f"score: {result.score:.4f}",
                        result.content,
                    ]
                )
            )
        return "\n\n".join(parts)

    async def ingest_documents(self, payload: IngestRequest) -> IngestResponse:
        """Read source documents, generate embeddings, and upsert vectors."""
        error = self._require_ingest_dependencies()
        if error:
            return IngestResponse(
                success=False,
                message=error,
                documents_discovered=0,
                chunks_indexed=0,
            )

        files = discover_documents(settings.rag_documents_dir)
        ensure_collection(force_recreate=payload.force_reindex)
        batch_size = max(1, settings.embedding_batch_size)

        chunks_indexed = 0
        failed_files: list[str] = []
        batch_texts: list[str] = []
        batch_points: list[PointStruct] = []

        async def flush_batch() -> None:
            nonlocal chunks_indexed, batch_texts, batch_points
            if not batch_texts:
                return
            for start in range(0, len(batch_texts), batch_size):
                texts = batch_texts[start : start + batch_size]
                points_slice = batch_points[start : start + batch_size]
                embeddings = await embed_texts(texts)
                points = [
                    PointStruct(
                        id=point.id,
                        vector=vector,
                        payload=point.payload,
                    )
                    for point, vector in zip(points_slice, embeddings, strict=True)
                ]
                upsert_points(points)
                chunks_indexed += len(points)
            batch_texts = []
            batch_points = []

        for path in files:
            try:
                chunks = chunk_document(path)
                if not payload.force_reindex:
                    indexed_chunks = count_points_for_source(str(path))
                    if indexed_chunks >= len(chunks):
                        continue
                for chunk_index, chunk in enumerate(chunks):
                    batch_texts.append(chunk)
                    batch_points.append(
                        PointStruct(
                            id=self._build_point_id(str(path), chunk_index),
                            vector=[],
                            payload={
                                "source": str(path),
                                "filename": path.name,
                                "chunk_index": chunk_index,
                                "content": chunk,
                            },
                        )
                    )
                    if len(batch_texts) >= batch_size:
                        await flush_batch()
            except Exception:
                failed_files.append(path.name)
        await flush_batch()

        message = "Document ingestion completed and vectors were upserted to Qdrant."
        if payload.force_reindex:
            message = "Force reindex completed and the collection was rebuilt."
        if failed_files:
            sample = ", ".join(failed_files[:3])
            message = (
                f"{message} Failed to parse {len(failed_files)} file(s): {sample}."
            )

        return IngestResponse(
            success=True,
            message=message,
            documents_discovered=len(files),
            chunks_indexed=chunks_indexed,
        )

    async def semantic_search(self, payload: SearchRequest) -> SearchResponse:
        """Run semantic retrieval."""
        error = self._require_ingest_dependencies()
        if error:
            return SearchResponse(query=payload.query, results=[])

        vector = (await embed_texts([payload.query]))[0]
        points = query_points(
            vector=vector,
            limit=payload.top_k or settings.top_k,
            score_threshold=settings.score_threshold,
        )
        results = [
            SearchResult(
                source=point.payload.get("source", ""),
                score=point.score or 0.0,
                content=point.payload.get("content", ""),
                metadata={
                    "filename": point.payload.get("filename"),
                    "chunk_index": point.payload.get("chunk_index"),
                },
            )
            for point in points
        ]
        return SearchResponse(query=payload.query, results=results)

    async def answer_question(self, payload: QueryRequest) -> QueryResponse:
        """Answer a question with RAG and fallback behavior."""
        if not payload.question.strip():
            raise ValueError("Question cannot be empty.")

        search_response = await self.semantic_search(
            SearchRequest(
                query=payload.question,
                top_k=payload.top_k or settings.top_k,
            )
        )
        strong_results = [
            result
            for result in search_response.results
            if result.score >= settings.score_threshold
        ]

        if strong_results:
            answer = await complete_chat(
                system_prompt=(
                    "You are a careful RAG assistant. Answer only from the provided "
                    "context. If the context is insufficient, say that directly."
                ),
                user_prompt=(
                    f"Question:\n{payload.question}\n\n"
                    f"Context:\n{self._format_context(strong_results)}\n\n"
                    "Write a concise answer in Chinese and mention the source file names "
                    "when useful."
                ),
            )
            return QueryResponse(
                question=payload.question,
                answer=answer.strip(),
                used_rag=True,
                fallback_used=False,
                sources=strong_results,
            )

        fallback_answer = await complete_chat(
            system_prompt=(
                "You are a helpful assistant. Answer the question directly. "
                "If you are uncertain, say so."
            ),
            user_prompt=payload.question,
        )
        return QueryResponse(
            question=payload.question,
            answer=fallback_answer.strip(),
            used_rag=False,
            fallback_used=True,
            sources=search_response.results,
        )

    async def get_stats(self) -> StatsResponse:
        """Return basic service stats."""
        files = discover_documents(settings.rag_documents_dir)
        return StatsResponse(
            service=settings.app_name,
            collection_name=settings.qdrant_collection_name,
            documents_dir=str(settings.rag_documents_dir),
            indexed_documents=len(files),
            indexed_chunks=count_points() if settings.qdrant_url and settings.qdrant_api_key else 0,
            embedding_model=settings.deepseek_embedding_model,
            chat_model=settings.deepseek_chat_model,
        )


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return a cached service instance."""
    return RAGService()
