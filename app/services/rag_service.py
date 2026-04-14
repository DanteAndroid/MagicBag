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
    ChatTurn,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.schemas.stats import StatsResponse
from app.services.document_loader import chunk_document, discover_documents, split_into_sentences


class RAGService:
    """Service object for RAG workflows."""

    MAX_HISTORY_TURNS = 5

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

    @classmethod
    def _focus_text(cls, question: str, history: list[ChatTurn]) -> str:
        recent_user_turns = [
            turn.content.strip()
            for turn in history[-(cls.MAX_HISTORY_TURNS * 2) :]
            if turn.role == "user" and turn.content.strip()
        ]
        return "\n".join([*recent_user_turns[-2:], question])

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

    @staticmethod
    def _is_summary_question(question: str) -> bool:
        lowered = question.lower()
        return any(
            hint in question or hint in lowered
            for hint in (
                "讲了什么",
                "主要讲什么",
                "内容是什么",
                "总结",
                "概述",
                "what is it about",
                "what's it about",
                "about what",
            )
        )

    @staticmethod
    def _is_followup_question(question: str) -> bool:
        lowered = question.lower().strip()
        return any(
            hint in question or hint in lowered
            for hint in (
                "它",
                "这本书",
                "这个方法",
                "这个流程",
                "这里面",
                "那",
                "那它",
                "那这",
                "what about",
                "how about",
                "and what",
            )
        )

    @staticmethod
    def _wants_multiple_items(question: str) -> bool:
        lowered = question.lower()
        return any(
            hint in question or hint in lowered
            for hint in (
                "哪些",
                "几种",
                "多个",
                "不同",
                "有哪些方法",
                "有哪些套路",
                "list",
                "multiple",
                "several",
                "different methods",
                "what methods",
                "which methods",
            )
        )

    @classmethod
    def _should_use_history(cls, question: str, history: list[ChatTurn]) -> bool:
        if not history:
            return False
        if cls._is_followup_question(question):
            return True

        recent_user_turns = [
            turn.content.strip()
            for turn in history[-(cls.MAX_HISTORY_TURNS * 2) :]
            if turn.role == "user" and turn.content.strip()
        ]
        if not recent_user_turns:
            return False

        latest_user_text = recent_user_turns[-1]
        lowered = latest_user_text.lower()
        return any(
            hint in latest_user_text or hint in lowered
            for hint in (
                "继续",
                "接着",
                "接下来",
                "这本书",
                "这个方法",
                "这个流程",
                "keep going",
                "continue",
                "follow up",
            )
        )

    @classmethod
    def _source_rank_info(cls, question: str, result: SearchResult) -> dict[str, float | bool]:
        title = str(result.metadata.get("title") or "")
        author = str(result.metadata.get("author") or "")
        filename = str(result.metadata.get("filename") or result.source)
        haystack = cls._normalized_text(" ".join([title, author, filename, result.content]))

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
        history: list[ChatTurn] | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        history = history or []
        focus_text = cls._focus_text(question, history) if cls._should_use_history(question, history) else question
        grouped = cls._merge_results_by_source(results, max_sources=None, max_snippets_per_source=1)
        ranked = []
        for result in grouped:
            rank_info = cls._source_rank_info(focus_text, result)
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
            if not cls._wants_multiple_items(question):
                if top["phrase_match"]:
                    if second is None or (not second["phrase_match"] and top["adjusted_score"] - second["adjusted_score"] >= 0.05):
                        select_count = 1
                    elif cls._is_method_question(question) and top["adjusted_score"] - second["adjusted_score"] >= 0.12:
                        select_count = 1
                    elif history and cls._is_followup_question(question) and top["adjusted_score"] - second["adjusted_score"] >= 0.04:
                        select_count = 1
                elif cls._is_method_question(question) and second is not None and top["adjusted_score"] - second["adjusted_score"] >= 0.22:
                    select_count = 1
            else:
                select_count = min(max(2, settings.max_context_sources), len(ranked))

        selected_keys = {item["key"] for item in ranked[:select_count]}
        if cls._wants_multiple_items(question) and cls._is_method_question(question):
            lexical_keys = {
                item["key"]
                for item in ranked
                if item["phrase_match"] or item["token_overlap"] >= 1
            }
            if lexical_keys:
                selected_keys &= lexical_keys
                if not selected_keys:
                    selected_keys = lexical_keys
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

    @classmethod
    def _compress_results_for_prompt(
        cls,
        question: str,
        history: list[ChatTurn],
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            return []

        focus_text = cls._focus_text(question, history)
        focus_tokens = cls._query_tokens(focus_text)
        focus_phrases = cls._query_phrases(focus_text)

        if cls._is_method_question(question):
            max_sentences = 10
            max_chars = 1800
        elif cls._is_summary_question(question):
            max_sentences = 8
            max_chars = 1500
        elif history and cls._is_followup_question(question):
            max_sentences = 6
            max_chars = 900
        else:
            max_sentences = 7
            max_chars = 1200

        compressed_results: list[SearchResult] = []
        for result in results:
            raw_sentences = split_into_sentences(result.content.replace("[Excerpt", "\n[Excerpt"))
            if not raw_sentences:
                compressed_results.append(result)
                continue

            scored_sentences: list[tuple[int, int, str]] = []
            for index, sentence in enumerate(raw_sentences):
                cleaned = re.sub(r"^\[Excerpt\s+\d+\]\s*", "", sentence).strip()
                haystack = cls._normalized_text(cleaned)
                token_hits = sum(1 for token in focus_tokens if token in haystack)
                phrase_hit = any(phrase in haystack for phrase in focus_phrases)
                score = token_hits * 2 + (3 if phrase_hit else 0)
                if index < 2:
                    score += 1
                scored_sentences.append((score, index, cleaned))

            scored_sentences.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            chosen = sorted(scored_sentences[:max_sentences], key=lambda item: item[1])

            content_parts: list[str] = []
            current_chars = 0
            for _, _, sentence in chosen:
                if not sentence or sentence in content_parts:
                    continue
                addition = len(sentence) + (2 if content_parts else 0)
                if content_parts and current_chars + addition > max_chars:
                    break
                content_parts.append(sentence)
                current_chars += addition

            compressed_results.append(
                SearchResult(
                    source=result.source,
                    score=result.score,
                    content="\n\n".join(content_parts) if content_parts else result.content,
                    metadata=dict(result.metadata),
                )
            )

        return compressed_results

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

    @classmethod
    def _history_text(cls, history: list[ChatTurn]) -> str:
        if not history:
            return ""

        recent_turns = history[-(cls.MAX_HISTORY_TURNS * 2) :]
        lines = [
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content.strip()}"
            for turn in recent_turns
            if turn.content.strip()
        ]
        if not lines:
            return ""
        return "Recent conversation:\n" + "\n".join(lines)

    @classmethod
    def _build_search_query(cls, question: str, history: list[ChatTurn]) -> str:
        if not cls._should_use_history(question, history):
            return question

        recent_user_turns = [
            turn.content.strip()
            for turn in history[-(cls.MAX_HISTORY_TURNS * 2) :]
            if turn.role == "user" and turn.content.strip()
        ]
        if not recent_user_turns:
            return question

        search_context = "\n".join(recent_user_turns[-2:])
        return f"{search_context}\nCurrent question: {question}"

    @classmethod
    def _answer_guidance(cls, question: str, language: str, history: list[ChatTurn]) -> str:
        followup_note = ""
        if cls._should_use_history(question, history):
            followup_note = (
                "Use the recent conversation to resolve references like 'it', 'this book', "
                "or 'that method' before answering.\n"
            )

        if cls._wants_multiple_items(question) and cls._is_method_question(question):
            if language == "zh":
                return (
                    followup_note
                    + "这是一个要求列举多个方法的问题。请按不同方法分别回答，每种方法单独成段，"
                    "写清名称、核心思路、流程特点和适用场景。如果上下文只提供部分细节，就明确标出哪一部分是已知信息。"
                )
            return (
                followup_note
                + "This question asks for multiple methods. Answer by separating distinct methods, "
                "and for each one explain its name, core idea, flow, and best use case."
            )

        if cls._is_method_question(question):
            if language == "zh":
                return (
                    followup_note
                    + "完整回答这个方法题。优先讲清：目标效果、准备条件、基本流程、关键细节、观众体验，"
                    "以及上下文里明确出现的限制或优点。不要只给高层概述。"
                )
            return (
                followup_note
                + "Answer this method question completely. Explain the effect, setup, flow, key details, "
                "audience experience, and any limitations or advantages stated in the context."
            )

        if cls._is_summary_question(question):
            if language == "zh":
                return (
                    followup_note
                    + "完整回答这类概述题：先说明核心主张，再展开至少 3 个更具体的观点、方法或判断。"
                    "不要只写空泛简介。"
                )
            return (
                followup_note
                + "Answer this summary question completely: state the central thesis first, then expand "
                "with at least 3 concrete ideas, methods, or judgments from the context."
            )

        if language == "zh":
            return (
                followup_note
                + "完整回答，不要敷衍。把上下文中的相关信息整合成一个有帮助的答案，不要让用户再去读来源。"
            )
        return (
            followup_note
            + "Answer completely and helpfully. Synthesize the relevant context into a direct answer "
            "instead of sending the user back to the sources."
        )

    @classmethod
    def _max_answer_tokens(cls, question: str, history: list[ChatTurn]) -> int:
        if cls._wants_multiple_items(question) and cls._is_method_question(question):
            return 900
        if cls._is_method_question(question):
            return 700
        if cls._is_summary_question(question):
            return 620
        if cls._should_use_history(question, history):
            return 420
        return 520

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
        if not payload.debug:
            return search_response

        selected_results = self._select_context_results(payload.query, search_response.results)
        expanded_results = self._expand_results_with_neighbors(selected_results)
        return SearchResponse(
            query=search_response.query,
            results=search_response.results,
            selected_results=selected_results,
            expanded_results=expanded_results,
        )

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
                query=self._build_search_query(payload.question, payload.history),
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
            selected_results = self._select_context_results(payload.question, strong_results, payload.history)
            expanded_results = self._expand_results_with_neighbors(selected_results)
            merged_results = self._compress_results_for_prompt(
                payload.question,
                payload.history,
                expanded_results,
            )
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
                    "context is insufficient, say that directly. Answer the question "
                    "completely and in detail based on the provided context. Never tell "
                    "the user to read the source; synthesize the information and answer directly. "
                    "If the context contains partial information across multiple passages, combine "
                    "the relevant pieces into one coherent answer."
                ),
                user_prompt=(
                    f"Question:\n{payload.question}\n\n"
                    f"{self._history_text(payload.history)}\n\n"
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
                    f"\n\nAnswer guidance:\n{self._answer_guidance(payload.question, payload.language, payload.history)}"
                    f"\n\nStyle guidance:\n{self._question_shape_guidance(payload.question, payload.language)}"
                    f"{self._relevant_term_guide(payload.language, payload.question, merged_results)}"
                ),
                max_tokens=self._max_answer_tokens(payload.question, payload.history),
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
                debug_results=search_response.results if payload.debug else [],
                debug_selected_results=selected_results if payload.debug else [],
                debug_expanded_results=expanded_results if payload.debug else [],
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
            user_prompt="\n\n".join(
                part for part in [self._history_text(payload.history), f"Question:\n{payload.question}"] if part
            ),
            max_tokens=360,
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
            debug_results=search_response.results if payload.debug else [],
            debug_selected_results=[],
            debug_expanded_results=[],
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
