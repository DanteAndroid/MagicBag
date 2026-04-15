"""DeepSeek-compatible API client helpers."""

from collections.abc import AsyncIterator

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


async def complete_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> str:
    """Generate a non-streaming chat completion."""
    client = get_deepseek_client()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    answer_parts: list[str] = []

    for _ in range(4):
        response = await client.chat.completions.create(
            model=settings.deepseek_chat_model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        chunk = choice.message.content or ""
        if chunk:
            answer_parts.append(chunk)
            messages.append({"role": "assistant", "content": chunk})

        if choice.finish_reason != "length":
            break

        messages.append(
            {
                "role": "user",
                "content": "Continue from exactly where you stopped. Do not repeat previous content.",
            }
        )

    return "".join(answer_parts).strip()


async def stream_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Generate a streaming chat completion."""
    client = get_deepseek_client()
    stream = await client.chat.completions.create(
        model=settings.deepseek_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        stream=True,
    )
    async for event in stream:
        if not event.choices:
            continue
        chunk = event.choices[0].delta.content or ""
        if chunk:
            yield chunk
