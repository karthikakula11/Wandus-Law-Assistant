"""Helpers to record usage from OpenAI chat completion objects."""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.monitoring.llm_usage_logger import record_llm_usage
from app.services.langfuse_tracing import get_request_session_id


async def record_chat_completion_usage(resp: Any, *, route: str) -> None:
    """Read ``usage`` + ``model`` from a non-streaming chat completion response."""
    u = getattr(resp, "usage", None)
    if u is None:
        return
    tin = int(getattr(u, "prompt_tokens", None) or 0)
    tout = int(getattr(u, "completion_tokens", None) or 0)
    model = getattr(resp, "model", None) or get_settings().openai_chat_model
    await record_llm_usage(
        model=model,
        tokens_in=tin,
        tokens_out=tout,
        session_id=get_request_session_id(),
        route=route,
    )


async def record_usage_dict(
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    route: str,
) -> None:
    """Record when only token counts are available (e.g. streaming final chunk)."""
    await record_llm_usage(
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        session_id=get_request_session_id(),
        route=route,
    )


async def record_embedding_usage(resp: Any, *, route: str = "embed-texts") -> None:
    """Record token usage from OpenAI ``embeddings.create`` (prompt_tokens only; no completion tokens)."""
    u = getattr(resp, "usage", None)
    if u is None:
        return
    tin = int(getattr(u, "prompt_tokens", None) or 0)
    if tin <= 0:
        return
    model = getattr(resp, "model", None) or get_settings().openai_embedding_model
    await record_llm_usage(
        model=model,
        tokens_in=tin,
        tokens_out=0,
        session_id=get_request_session_id(),
        route=route,
    )
