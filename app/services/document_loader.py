"""Document discovery and parsing helpers.

Implement real file reading, markdown/pdf/docx parsing, and chunking here.
The current version only discovers files so the skeleton runs without extra
business rules.
"""

from pathlib import Path


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


def chunk_document(_: Path) -> list[str]:
    """Split a source document into chunks.

    TODO: Replace this placeholder with actual file parsing and semantic/text
    chunking logic based on `chunk_size` and `chunk_overlap`.
    """
    return []
