"""DeepSeek-compatible API client helpers."""

from openai import AsyncOpenAI

from app.core.config import settings


def get_deepseek_client() -> AsyncOpenAI:
    """Return an async DeepSeek client."""
    return AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for a batch of texts."""
    if not texts:
        return []

    response = await get_deepseek_client().embeddings.create(
        input=texts,
        model=settings.deepseek_embedding_model,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]


async def complete_chat(system_prompt: str, user_prompt: str) -> str:
    """Generate a non-streaming chat completion."""
    response = await get_deepseek_client().chat.completions.create(
        model=settings.deepseek_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""
