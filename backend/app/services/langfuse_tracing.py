"""
Langfuse tracing helpers (aligned with Langfuse skill / instrumentation best practices).

- Request-scoped session id is stored in a context var and attached to LangGraph via
  ``langfuse_session_id`` in runnable metadata (CallbackHandler). Do **not** pass
  ``session_id`` / ``user_id`` into ``chat.completions.create`` — Langfuse's OpenAI wrapper
  does not strip them from kwargs, and the real OpenAI API rejects unknown arguments.
- OpenAI calls add only ``name`` for the Langfuse OpenAI integration when keys are set.

Import this module only after settings/env are loaded (see skill: Langfuse before OpenAI).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Iterator

from app.monitoring.langfuse_config import get_langfuse_config, langfuse_enabled

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

_lf_session_id: ContextVar[str | None] = ContextVar("langfuse_session_id", default=None)


def is_langfuse_configured() -> bool:
    """True when public key, secret key, and host are configured (see ``langfuse_config``)."""
    return langfuse_enabled()


def get_request_session_id() -> str | None:
    return _lf_session_id.get()


@contextmanager
def lf_request_context(session_id: str | None) -> Iterator[None]:
    """Bind a Langfuse session id for the current request (chat turn)."""
    if not session_id:
        yield
        return
    token = _lf_session_id.set(session_id)
    try:
        yield
    finally:
        _lf_session_id.reset(token)


def openai_trace_kwargs(*, name: str) -> dict[str, Any]:
    """
    Extra kwargs for ``chat.completions.create`` when Langfuse wrapping is active.

    Only ``name`` is safe to pass: Langfuse reads it via ``OpenAiArgsExtractor`` and omits
    it from the forwarded kwargs. ``session_id`` / ``user_id`` must not be passed here —
    they remain in ``kwargs`` and are forwarded to the real OpenAI client, which errors.
    Session for traces comes from ``build_langgraph_run_config`` metadata instead.
    """
    if not is_langfuse_configured():
        return {}
    return {"name": name}


def build_langgraph_run_config(
    base: "RunnableConfig",
    *,
    trace_name: str = "pintu-agentic-rag",
) -> "RunnableConfig":
    """
    Merge Langfuse ``CallbackHandler``, ``run_name``, and metadata for LangGraph.

    Uses ``langfuse_session_id`` / tags expected by Langfuse's LangChain handler.
    """
    if not is_langfuse_configured():
        return base

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as e:
        # Optional: install ``langchain`` if you want LangGraph runs in Langfuse UI
        # (``pip install langchain`` — resolve version with ``langchain-core`` / LangGraph).
        logger.debug("langfuse.langchain CallbackHandler unavailable: %s", e)
        return base

    handler = CallbackHandler()
    meta: dict[str, Any] = {
        "langfuse_trace_name": trace_name,
        "langfuse_tags": ["pintu", "law-chatbot", "agentic-rag"],
    }
    sid = get_request_session_id()
    if sid:
        meta["langfuse_session_id"] = sid

    out: dict[str, Any] = dict(base)
    out["callbacks"] = [handler]
    out["metadata"] = meta
    out["run_name"] = trace_name
    return out


def flush_langfuse() -> None:
    """Best-effort flush of queued Langfuse events (e.g. before process exit)."""
    if not is_langfuse_configured():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        logger.debug("langfuse flush skipped: %s", e)


def init_langfuse_client_from_settings() -> None:
    """Register the Langfuse SDK with explicit credentials before any OpenAI call.

    The OpenAI wrapper resolves ``get_client(public_key=None)`` → ``Langfuse()``, which
    relies on environment variables. We already sync env in
    ``apply_langfuse_env_from_settings``; constructing ``Langfuse(public_key=..., ...)``
    here ensures ``LangfuseResourceManager`` registers this project so tracing is not a
    no-op and OTLP export is wired.
    """
    if not langfuse_enabled():
        return
    cfg = get_langfuse_config()
    if cfg is None:
        return
    try:
        from langfuse import Langfuse

        Langfuse(
            public_key=cfg.public_key,
            secret_key=cfg.secret_key,
            base_url=cfg.base_url,
        )
        logger.info(
            "Langfuse tracing ready (host=%s, public_key=%s…)",
            cfg.base_url,
            cfg.public_key[:12],
        )
    except Exception as e:
        logger.warning("Langfuse client init failed (traces may be missing): %s", e)
