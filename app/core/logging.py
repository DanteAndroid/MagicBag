"""Logging setup.

Replace this with structured logging / tracing if needed later.
"""

import logging

from app.core.config import settings


def configure_logging() -> None:
    """Initialize process-wide logging once on startup."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
