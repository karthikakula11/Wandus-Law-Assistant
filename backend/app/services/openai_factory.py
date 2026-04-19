"""Return ``AsyncOpenAI`` — Langfuse-wrapped when Langfuse keys are set, else plain OpenAI.

All OpenAI usage in this app (chat, embeddings, streaming) must go through
``get_async_openai_client()`` so Langfuse can observe generations/embeddings when
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are set. Do not import ``AsyncOpenAI``
from ``openai`` elsewhere in ``app/``.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache
def get_async_openai_client():  # noqa: ANN201
    """Shared async OpenAI client (one instance per process)."""
    settings = get_settings()
    key = settings.openai_api_key
    if (settings.langfuse_public_key or "").strip() and (settings.langfuse_secret_key or "").strip():
        from langfuse.openai import AsyncOpenAI

        return AsyncOpenAI(api_key=key)
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=key)


@lru_cache
def get_async_ollama_openai_client():  # noqa: ANN201
    """
    OpenAI-compatible client for **Ollama** (``/v1/chat/completions``).

    Not Langfuse-wrapped; used for optional local calls (e.g. auto-memory extraction).
    """
    from openai import AsyncOpenAI

    settings = get_settings()
    base = settings.ollama_base_url.strip().rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return AsyncOpenAI(api_key="ollama", base_url=base)


def clear_openai_client_cache() -> None:
    get_async_openai_client.cache_clear()
    get_async_ollama_openai_client.cache_clear()
