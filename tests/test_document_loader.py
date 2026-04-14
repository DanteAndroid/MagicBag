from pathlib import Path

from app.services.document_loader import chunk_document, chunk_text, discover_documents, split_into_sentences


def test_discover_documents_filters_supported_files(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("hello", encoding="utf-8")
    (tmp_path / "ignore.png").write_text("x", encoding="utf-8")

    discovered = discover_documents(tmp_path)

    assert [path.name for path in discovered] == ["note.md"]


def test_split_into_sentences_preserves_boundaries() -> None:
    sentences = split_into_sentences("第一句。第二句！Third sentence. Fourth sentence?")
    assert sentences == ["第一句。", "第二句！", "Third sentence.", "Fourth sentence?"]


def test_chunk_text_uses_sentence_boundaries_and_overlap() -> None:
    text = "第一句很短。第二句也很短。第三句稍微长一点。第四句用于测试。"

    chunks = chunk_text(text, chunk_size=18, chunk_overlap=8)

    assert chunks == [
        "第一句很短。 第二句也很短。",
        "第二句也很短。 第三句稍微长一点。",
        "第三句稍微长一点。 第四句用于测试。",
    ]


def test_chunk_text_keeps_long_sentence_intact() -> None:
    text = "This is a very long sentence without a natural place to split even though it exceeds the target size."
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert chunks == [text]


def test_chunk_document_reads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nhello world", encoding="utf-8")

    chunks = chunk_document(path)

    assert chunks
    assert "hello world" in chunks[0]
