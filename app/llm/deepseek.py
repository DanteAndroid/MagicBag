"""DeepSeek API client wrapper.

Both embedding and chat completion placeholders live here. The skeleton uses
OpenAI-compatible requests because DeepSeek exposes a similar API style.
"""

from openai import AsyncOpenAI

from app.core.config import settings


def get_deepseek_client() -> AsyncOpenAI:
    """Return an async DeepSeek client."""
    return AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
