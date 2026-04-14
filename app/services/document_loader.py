"""Document discovery and parsing helpers."""

from pathlib import Path
import re

from docx import Document
from pypdf import PdfReader

from app.core.config import settings


SUPPORTED_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
}


def discover_documents(base_dir: Path) -> list[Path]:
    """Recursively discover supported source documents."""
    if not base_dir.exists():
        return []

    return sorted(
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_document(path: Path) -> str:
    """Read a supported document into plain text."""
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        raw = re.sub(r"<(script|style).*?>.*?</\\1>", " ", raw, flags=re.I | re.S)
        raw = re.sub(r"<[^>]+>", " ", raw)
        return raw

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    raise ValueError(f"Unsupported document type: {path.suffix}")


def normalize_text(text: str) -> str:
    """Normalize whitespace before chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like units while preserving sentence endings."""
    if not text:
        return []

    blocks = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    sentence_pattern = re.compile(r".+?(?:[。！？!?]+|\.{3,}|…{1,2}|(?<!\d)\.(?!\d)|$)", re.S)

    for block in blocks:
        stripped_block = block.strip()
        if not stripped_block:
            continue

        for match in sentence_pattern.finditer(stripped_block):
            sentence = match.group(0).strip()
            if sentence:
                sentences.append(sentence)

    return sentences


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping sentence-aligned chunks."""
    text = normalize_text(text)
    if not text:
        return []

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    index = 0

    while index < len(sentences):
        current_sentences: list[str] = []
        current_length = 0
        cursor = index

        while cursor < len(sentences):
            sentence = sentences[cursor]
            separator = 1 if current_sentences else 0
            proposed_length = current_length + separator + len(sentence)
            if current_sentences and proposed_length > size:
                break
            current_sentences.append(sentence)
            current_length = proposed_length
            cursor += 1

            # Allow a single sentence to exceed the target size rather than hard-cutting it.
            if len(current_sentences) == 1 and len(sentence) >= size:
                break

        if not current_sentences:
            break

        chunks.append(" ".join(current_sentences).strip())
        if cursor >= len(sentences):
            break

        if overlap == 0:
            index = cursor
            continue

        overlap_length = 0
        next_index = cursor
        while next_index > index:
            sentence = sentences[next_index - 1]
            separator = 1 if overlap_length else 0
            proposed_overlap = overlap_length + separator + len(sentence)
            if overlap_length and proposed_overlap > overlap:
                break
            overlap_length = proposed_overlap
            next_index -= 1

        if next_index == index:
            next_index = min(cursor, index + 1)
        index = next_index

    return chunks


def chunk_document(path: Path) -> list[str]:
    """Read and split a source document into chunks."""
    return chunk_text(
        read_document(path),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
