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


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping character chunks."""
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

    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        if chunks and start + overlap >= len(text):
            break
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_document(path: Path) -> list[str]:
    """Read and split a source document into chunks."""
    return chunk_text(
        read_document(path),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
