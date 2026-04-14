"""RAG service orchestration.

This is the main place to implement ingestion, retrieval, reranking, prompt
assembly, and LLM fallback behavior.
"""

from functools import lru_cache
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid5, NAMESPACE_URL

from qdrant_client.models import PointStruct

from app.core.config import settings
from app.db.qdrant import (
    count_points,
    count_points_for_source,
    ensure_collection,
    fetch_source_chunks,
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

    QUERY_SYNONYMS = {
        "迫选": "Force",
        "迫牌": "Forcing card",
        "自由选择": "Free choice",
        "双关语": "Equivoque",
        "控牌": "Control",
        "双翻": "Double lift",
        "间隔": "Break",
        "切牌": "Cut",
        "洗牌": "Shuffle",
        "预设顺序": "Stack",
        "偷看": "Peek",
        "瞬间偷看": "Glimpse",
        "印记器": "Impression device",
        "中心撕纸法": "Center tear",
        "纸条": "Billet",
        "偷换": "Switch",
        "指写器": "Nail writer",
        "揭示": "Revelation",
        "预言": "Prediction",
        "心中所想的牌": "Thought-of card",
        "暗示": "Suggestion",
        "心理强迫": "Psychological force",
        "语言控牌": "Verbal control",
        "框架": "Framing",
        "引导": "Leading",
        "顺从": "Compliance",
        "错误引导": "Misdirection",
        "注意力管理": "Attention management",
        "可信动作": "Convincer",
        "手法": "Sleight",
        "纯手法": "Sleight of hand",
        "藏牌": "Palm",
        "换顶牌": "Top change",
        "假洗": "False shuffle",
        "假切": "False cut",
        "关键牌": "Key card",
        "折痕标记": "Crimp",
        "记号牌": "Marked deck",
        "流程": "Routine",
        "效果": "Effect",
        "开场效果": "Opener",
        "收尾效果": "Closer",
        "呼应效果": "Callback",
        "节奏点": "Beat",
        "时机": "Timing",
        "干净": "Clean",
    }

    TERM_GLOSSARY = {
        "Force": "迫选",
        "Forcing card": "迫牌",
        "Free choice": "自由选择",
        "Equivoque": "双关语",
        "Control": "控牌",
        "Double lift": "双翻",
        "Break": "间隔",
        "Cut": "切牌",
        "Shuffle": "洗牌",
        "Stack": "预设顺序",
        "Peek": "偷看",
        "Glimpse": "瞬间偷看",
        "Impression device": "印记器",
        "Center tear": "中心撕纸法",
        "Billet": "纸条",
        "Switch": "偷换",
        "Nail writer": "指写器",
        "Revelation": "揭示",
        "Prediction": "预言",
        "Thought-of card": "心中所想的牌",
        "Suggestion": "暗示",
        "Psychological force": "心理强迫",
        "Verbal control": "语言控牌",
        "Framing": "框架",
        "Leading": "引导",
        "Compliance": "顺从",
        "Misdirection": "错误引导",
        "Attention management": "注意力管理",
        "Convincer": "可信动作",
        "Sleight": "手法",
        "Sleight of hand": "纯手法",
        "Pass": "Pass",
        "Palm": "藏牌",
        "Top change": "换顶牌",
        "False shuffle": "假洗",
        "False cut": "假切",
        "Key card": "关键牌",
        "Crimp": "折痕标记",
        "Marked deck": "记号牌",
        "Routine": "流程",
        "Effect": "效果",
        "Opener": "开场效果",
        "Closer": "收尾效果",
        "Callback": "呼应效果",
        "Beat": "节奏点",
        "Timing": "时机",
        "Clean": "干净",
    }

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
            title = result.metadata.get("title")
            author = result.metadata.get("author")
            parts.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"file: {result.metadata.get('filename') or result.source}",
                        f"title: {title or 'unknown'}",
                        f"author: {author or 'unknown'}",
                        f"score: {result.score:.4f}",
                        result.content,
                    ]
                )
            )
        return "\n\n".join(parts)

    @staticmethod
    def _language_name(language: str) -> str:
        return "English" if language == "en" else "Chinese"

    @staticmethod
    def _source_key(result: SearchResult) -> str:
        return str(result.metadata.get("filename") or result.source)

    @classmethod
    def _normalized_text(cls, text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text.lower()).strip()

    @classmethod
    def _query_phrases(cls, question: str) -> list[str]:
        phrases = [
            cls._normalized_text(match.group(0))
            for match in re.finditer(r"[A-Za-z][A-Za-z0-9'&-]*(?:\s+[A-Za-z][A-Za-z0-9'&-]*)*", question)
        ]
        return [phrase for phrase in phrases if phrase and (len(phrase.split()) >= 2 or len(phrase) >= 5)]

    @classmethod
    def _query_tokens(cls, question: str) -> list[str]:
        tokens = cls._normalized_text(question).split()
        return [token for token in tokens if len(token) >= 4]

    @staticmethod
    def _is_method_question(question: str) -> bool:
        lowered = question.lower()
        return any(
            hint in question or hint in lowered
            for hint in (
                "方法",
                "做法",
                "怎么做",
                "步骤",
                "流程",
                "手顺",
                "细节",
                "handling",
                "method",
                "methods",
                "how to",
                "how do",
                "routine",
            )
        )

    @classmethod
    def _source_rank_info(cls, question: str, result: SearchResult) -> dict[str, float | bool]:
        title = str(result.metadata.get("title") or "")
        author = str(result.metadata.get("author") or "")
        filename = str(result.metadata.get("filename") or result.source)
        haystack = cls._normalized_text(" ".join([title, author, filename]))

        phrases = cls._query_phrases(question)
        tokens = cls._query_tokens(question)
        phrase_match = any(phrase in haystack for phrase in phrases)
        token_overlap = sum(1 for token in tokens if token in haystack)

        lexical_bonus = 0.0
        if phrase_match:
            lexical_bonus += 0.8
        lexical_bonus += min(token_overlap * 0.12, 0.36)

        return {
            "phrase_match": phrase_match,
            "token_overlap": float(token_overlap),
            "lexical_bonus": lexical_bonus,
        }

    @classmethod
    def _select_context_results(
        cls,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            return []

        grouped = cls._merge_results_by_source(results, max_sources=None, max_snippets_per_source=1)
        ranked = []
        for result in grouped:
            rank_info = cls._source_rank_info(question, result)
            ranked.append(
                {
                    "key": cls._source_key(result),
                    "adjusted_score": result.score + float(rank_info["lexical_bonus"]),
                    "phrase_match": bool(rank_info["phrase_match"]),
                    "token_overlap": float(rank_info["token_overlap"]),
                }
            )

        ranked.sort(key=lambda item: item["adjusted_score"], reverse=True)
        select_count = settings.max_context_sources
        if ranked:
            top = ranked[0]
            second = ranked[1] if len(ranked) > 1 else None
            if top["phrase_match"]:
                if second is None or (not second["phrase_match"] and top["adjusted_score"] - second["adjusted_score"] >= 0.05):
                    select_count = 1
                elif cls._is_method_question(question) and top["adjusted_score"] - second["adjusted_score"] >= 0.12:
                    select_count = 1
            elif cls._is_method_question(question) and second is not None and top["adjusted_score"] - second["adjusted_score"] >= 0.22:
                select_count = 1

        selected_keys = {item["key"] for item in ranked[:select_count]}
        selected_results = [result for result in results if cls._source_key(result) in selected_keys]
        selected_results.sort(key=lambda item: item.score, reverse=True)
        return selected_results

    @classmethod
    def _merge_chunk_ranges(cls, chunk_indexes: list[int], window: int) -> list[tuple[int, int]]:
        if not chunk_indexes:
            return []

        ranges = sorted((max(0, index - window), index + window) for index in chunk_indexes)
        merged = [ranges[0]]
        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @classmethod
    def _merge_results_by_source(
        cls,
        results: list[SearchResult],
        *,
        max_sources: int | None = None,
        max_snippets_per_source: int = 3,
    ) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        snippets_seen: dict[str, list[str]] = {}

        for result in results:
            key = cls._source_key(result)
            snippet = result.content.strip()
            if key not in merged:
                merged[key] = SearchResult(
                    source=result.source,
                    score=result.score,
                    content="",
                    metadata=dict(result.metadata),
                )
                snippets_seen[key] = []

            if snippet and snippet not in snippets_seen[key]:
                snippets_seen[key].append(snippet)
                collected = snippets_seen[key][:max_snippets_per_source]
                merged[key].content = "\n\n".join(
                    f"[Excerpt {index}] {text}"
                    for index, text in enumerate(collected, start=1)
                )

            if result.score > merged[key].score:
                merged[key].score = result.score

        merged_results = list(merged.values())
        merged_results.sort(key=lambda item: item.score, reverse=True)
        if max_sources is not None:
            return merged_results[:max_sources]
        return merged_results

    @classmethod
    def _expand_results_with_neighbors(
        cls,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            return []

        merged_results = cls._merge_results_by_source(
            results,
            max_sources=settings.max_context_sources,
            max_snippets_per_source=settings.max_chunks_per_source,
        )
        expanded_results: list[SearchResult] = []

        for result in merged_results:
            source = result.source
            raw_indexes = [
                item.metadata.get("chunk_index")
                for item in results
                if item.source == source and isinstance(item.metadata.get("chunk_index"), int)
            ]
            chunk_indexes = sorted(set(int(index) for index in raw_indexes if index is not None))
            ranges = cls._merge_chunk_ranges(chunk_indexes, settings.query_chunk_window)

            excerpts: list[str] = []
            for start, end in ranges:
                for record in fetch_source_chunks(source, start, end):
                    content = str(record.payload.get("content", "")).strip()
                    if content and content not in excerpts:
                        excerpts.append(content)
                    if len(excerpts) >= settings.max_chunks_per_source:
                        break
                if len(excerpts) >= settings.max_chunks_per_source:
                    break

            if excerpts:
                result.content = "\n\n".join(
                    f"[Excerpt {index}] {text}"
                    for index, text in enumerate(excerpts[: settings.max_chunks_per_source], start=1)
                )
            expanded_results.append(result)

        return expanded_results

    @staticmethod
    def _is_author_candidate(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned or any(char.isdigit() for char in cleaned):
            return False

        words = [word for word in cleaned.replace("’", "'").split() if word]
        if not words or len(words) > 5:
            return False

        if words[0].lower() in {"the", "a", "an"}:
            return False

        capitalized = 0
        for word in words:
            token = word.strip(".,()[]{}'\"")
            if not token:
                continue
            if token[0].isupper():
                capitalized += 1

        return capitalized >= max(1, len(words) - 1)

    @classmethod
    def _reference_metadata(cls, source: str, filename: str | None) -> dict[str, str | int | float | bool | None]:
        resolved_filename = filename or Path(source).name
        stem = Path(resolved_filename).stem.strip()
        title = stem
        author: str | None = None

        if " - " in stem:
            left, right = stem.split(" - ", 1)
            left = left.strip()
            right = right.strip()
            if cls._is_author_candidate(left):
                author = left
                title = right or title
            elif cls._is_author_candidate(right):
                title = left or title
                author = right
        else:
            stripped = re.sub(r"^\d+\s+", "", stem).strip()
            words = stripped.split()
            if len(words) == 4:
                candidate_author = " ".join(words[-2:])
                candidate_title = " ".join(words[:-2]).strip()
                if candidate_title and cls._is_author_candidate(candidate_author):
                    author = candidate_author
                    title = candidate_title

        return {
            "filename": resolved_filename,
            "title": title,
            "author": author,
        }

    @classmethod
    def _expand_query(cls, query: str) -> str:
        """Append canonical English terms for known Chinese glossary entries."""
        additions: list[str] = []
        lowered = query.lower()
        for zh_term, canonical in cls.QUERY_SYNONYMS.items():
            if zh_term in query and canonical.lower() not in lowered:
                additions.append(canonical)
        if not additions:
            return query
        return f"{query}\n\nRelated glossary terms: {', '.join(additions)}"

    @classmethod
    def _term_guide(cls, language: str) -> str:
        if language != "zh":
            return ""

        return ""

    @classmethod
    def _relevant_term_guide(cls, language: str, question: str, results: list[SearchResult]) -> str:
        """Return only glossary lines relevant to the current question/context."""
        if language != "zh":
            return ""

        haystack = "\n".join(
            [question, *[result.content for result in results], *[result.source for result in results]]
        ).lower()
        matched: list[str] = []
        for zh_term, canonical in cls.QUERY_SYNONYMS.items():
            if zh_term in haystack or canonical.lower() in haystack:
                matched.append(f"{canonical} -> {cls.TERM_GLOSSARY.get(canonical, zh_term)}")

        if not matched:
            return ""

        unique_lines = list(dict.fromkeys(matched))[:8]
        return (
            "\n\nTerminology guide for Chinese answers:\n"
            + "\n".join(unique_lines)
            + "\nIf one of these terms is needed, prefer the glossary wording."
        )

    @staticmethod
    def _question_shape_guidance(question: str, language: str) -> str:
        lowered = question.lower()
        if any(token in question for token in ("讲了什么", "主要讲什么", "内容是什么", "总结一下")) or (
            "what" in lowered and "about" in lowered
        ):
            if language == "zh":
                return (
                    "先用一两句话概括核心观点，再补充 3 个最重要的主题、方法或启发。"
                    "重点讲内容本身，不要把回答写成书目信息。"
                )
            return (
                "Start with a concise summary, then expand with 3 key ideas, methods, "
                "or takeaways. Focus on the content, not bibliography."
            )

        if language == "zh":
            return (
                "默认给出较充实的回答：先直接回答问题，再补充 2 到 4 个关键点或例子。"
                "不要只给一小段空泛概述。"
            )
        return (
            "Give a substantive answer by default: answer directly, then add 2 to 4 key "
            "points or examples. Avoid a thin generic paragraph."
        )

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
        search_response, _, _ = await self._semantic_search_with_timings(payload)
        return search_response

    async def _semantic_search_with_timings(
        self,
        payload: SearchRequest,
    ) -> tuple[SearchResponse, int, int]:
        """Run semantic retrieval and capture embedding/search timings."""
        error = self._require_ingest_dependencies()
        if error:
            return SearchResponse(query=payload.query, results=[]), 0, 0

        expanded_query = self._expand_query(payload.query)
        embedding_started = perf_counter()
        vector = (await embed_texts([expanded_query]))[0]
        embedding_time_ms = int((perf_counter() - embedding_started) * 1000)
        vector_search_started = perf_counter()
        points = query_points(
            vector=vector,
            limit=payload.top_k or settings.top_k,
            score_threshold=settings.score_threshold,
        )
        vector_search_time_ms = int((perf_counter() - vector_search_started) * 1000)
        results = [
            SearchResult(
                source=point.payload.get("source", ""),
                score=point.score or 0.0,
                content=point.payload.get("content", ""),
                metadata=(
                    self._reference_metadata(
                        point.payload.get("source", ""),
                        point.payload.get("filename"),
                    )
                    | {
                        "chunk_index": point.payload.get("chunk_index"),
                    }
                ),
            )
            for point in points
        ]
        return (
            SearchResponse(query=payload.query, results=results),
            embedding_time_ms,
            vector_search_time_ms,
        )

    async def answer_question(self, payload: QueryRequest) -> QueryResponse:
        """Answer a question with RAG and fallback behavior."""
        if not payload.question.strip():
            raise ValueError("Question cannot be empty.")

        request_started = perf_counter()
        search_response, embedding_time_ms, vector_search_time_ms = await self._semantic_search_with_timings(
            SearchRequest(
                query=payload.question,
                top_k=payload.top_k or settings.top_k,
            )
        )
        retrieval_time_ms = embedding_time_ms + vector_search_time_ms
        strong_results = [
            result
            for result in search_response.results
            if result.score >= settings.score_threshold
        ]

        if strong_results:
            answer_language = self._language_name(payload.language)
            selected_results = self._select_context_results(payload.question, strong_results)
            merged_results = self._expand_results_with_neighbors(selected_results)
            generation_started = perf_counter()
            answer = await complete_chat(
                system_prompt=(
                    "You are a careful RAG assistant for magic teaching and performance study. "
                    "Assume the user is asking about lawful entertainment, stagecraft, "
                    "or consensual performance for magicians. Do not moralize about "
                    "magic methods. Use only the provided context and do not use "
                    "outside knowledge or memory. If a title, author, date, or term "
                    "appears in the context, preserve it exactly as written there. "
                    "Treat structured source metadata such as file, title, and author "
                    "as authoritative. If author is unknown in the context, do not "
                    "guess or substitute one. "
                    "Do not mention internal retrieval mechanics, vector databases, file "
                    "storage, chunking, or phrases like 'according to the file/context' "
                    "unless the user explicitly asks about sources or system behavior. "
                    "Do not translate or rewrite proper nouns unless the translated "
                    "form is already present in the context. Never introduce a person, "
                    "book, method, or recommendation that does not appear in the "
                    "context. Never merge two different names into one. If your prior "
                    "knowledge conflicts with the context, trust the context. If the "
                    "context is insufficient, say that directly."
                ),
                user_prompt=(
                    f"Question:\n{payload.question}\n\n"
                    f"Context:\n{self._format_context(merged_results)}\n\n"
                    f"Write a direct, natural answer in {answer_language} for a magician "
                    "studying performance methods. The UI already shows the references "
                    "below the answer, so do not mention raw file names, internal files, "
                    "or say 'according to the file/context/database'. Answer only the "
                    "user's question and do not add extra recommendations "
                    "or related people unless explicitly asked. Do not invent or substitute "
                    "a different author, title, or book. If the context names the source "
                    "file, title, or author, prefer those exact values. If the context does "
                    "not provide an author, say that the provided excerpt does not identify "
                    "the author instead of guessing. If the answer language is Chinese, "
                    "use natural, idiomatic Simplified Chinese instead of literal translation, "
                    "but keep names and titles exactly as they appear in the context. "
                    "When the context contains method steps, procedure details, or handling "
                    "nuance, include those practical details instead of only giving a high-level summary. "
                    "The context may be long because this is a magic teaching workflow; extract the most useful details and explain them clearly."
                    f"\n\nStyle guidance:\n{self._question_shape_guidance(payload.question, payload.language)}"
                    f"{self._relevant_term_guide(payload.language, payload.question, merged_results)}"
                ),
            )
            generation_time_ms = int((perf_counter() - generation_started) * 1000)
            return QueryResponse(
                question=payload.question,
                answer=answer.strip(),
                used_rag=True,
                fallback_used=False,
                sources=merged_results,
                embedding_time_ms=embedding_time_ms,
                vector_search_time_ms=vector_search_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_time_ms=int((perf_counter() - request_started) * 1000),
            )

        generation_started = perf_counter()
        fallback_answer = await complete_chat(
            system_prompt=(
                "You are a helpful assistant for magic performance study. Assume the "
                "user is asking about lawful entertainment, stagecraft, or consensual "
                "performance for magicians. Do not refuse simply because a method "
                "involves deception as part of stage magic. Answer directly in "
                f"{self._language_name(payload.language)} and, if uncertain, say so."
            ),
            user_prompt=payload.question,
        )
        generation_time_ms = int((perf_counter() - generation_started) * 1000)
        return QueryResponse(
            question=payload.question,
            answer=fallback_answer.strip(),
            used_rag=False,
            fallback_used=True,
            sources=search_response.results,
            embedding_time_ms=embedding_time_ms,
            vector_search_time_ms=vector_search_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=int((perf_counter() - request_started) * 1000),
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
