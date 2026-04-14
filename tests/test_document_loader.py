from pathlib import Path

from app.services.document_loader import chunk_document, chunk_text, discover_documents


def test_discover_documents_filters_supported_files(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("hello", encoding="utf-8")
    (tmp_path / "ignore.png").write_text("x", encoding="utf-8")

    discovered = discover_documents(tmp_path)

    assert [path.name for path in discovered] == ["note.md"]


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_document_reads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nhello world", encoding="utf-8")

    chunks = chunk_document(path)

    assert chunks
    assert "hello world" in chunks[0]
