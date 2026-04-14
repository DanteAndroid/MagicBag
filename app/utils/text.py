"""Text utility helpers.

Add normalization, token counting, or prompt formatting helpers here as your
RAG logic grows.
"""


def normalize_text(value: str) -> str:
    """Normalize user input or chunk text."""
    return " ".join(value.split())
