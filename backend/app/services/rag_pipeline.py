"""
Course-style RAG pipeline phases (jamwithai/production-agentic-rag-course):
retrieve (handled in `rag.retrieve_for_chat`) → prompt build → LLM generate.

This module holds the **generation phase** only to keep a single place for
`RAGPromptBuilder` + async LLM injection (sync or stream).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.models import Chunk, Document
from app.services.rag_prompts import RAGPromptBuilder

T = TypeVar("T")


async def run_generation_phase(
    question: str,
    contexts: list[tuple[Chunk, Document]],
    history: list[dict],
    llm: Callable[[list[dict]], Awaitable[T]],
    *,
    memory_snippets: list[str] | None = None,
) -> tuple[T, list[tuple[Chunk, Document]]]:
    """
    Build messages with RAGPromptBuilder, then call `llm(messages)`.
    `llm` may return a str (non-streaming) or an async iterator (if you wrap streaming).
    """
    messages, included = RAGPromptBuilder().build_messages(
        question, contexts, history, memory_snippets=memory_snippets
    )
    out = await llm(messages)
    return out, included
