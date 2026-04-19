"""Offline benchmark helpers: basic (single-query dense) vs agentic LangGraph retrieval."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.services.rag import retrieve_chunks
from app.services.rag_graph import accumulate_agentic_state_with_trace

logger = logging.getLogger(__name__)


def _rows_to_chunk_ids(
    rows: list[tuple[Any, Any, float]],
) -> list[UUID]:
    out: list[UUID] = []
    seen: set[UUID] = set()
    for ch, _doc, _d in rows:
        if ch.id not in seen:
            seen.add(ch.id)
            out.append(ch.id)
    return out


async def run_basic_retrieval(
    session: AsyncSession,
    question: str,
    top_k: int,
    *,
    scope_document_ids: list[UUID] | None = None,
) -> list[UUID]:
    """Baseline: one embedding of the user question → dense top‑k (no plan/rewrite loop)."""
    rows = await retrieve_chunks(
        session, question.strip(), top_k, scope_document_ids=scope_document_ids
    )
    return _rows_to_chunk_ids(rows)


async def run_agentic_retrieval_and_trace(
    session: AsyncSession,
    question: str,
    top_k: int,
    *,
    scope_document_ids: list[UUID] | None = None,
) -> tuple[list[UUID], list[str]]:
    """Agentic path: LangGraph (plan → retrieve → grade → … → generate). Uses final ``rows``."""
    acc, trace = await accumulate_agentic_state_with_trace(
        session,
        question.strip(),
        top_k,
        history=None,
        scope_document_ids=scope_document_ids,
        memory_snippets=[],
    )
    rows = acc.get("rows") or []
    ids = _rows_to_chunk_ids(rows)
    return ids, trace


async def resolve_gold_chunk_ids(
    session: AsyncSession,
    item: dict[str, Any],
) -> list[UUID]:
    """From benchmark item: explicit UUIDs and/or document title + chunk_index.

    If ``gold_auto`` is true, use the first chunk of a document whose title matches
    ``gold_document_title`` (ilike), or the first chunk in the DB — for demos when labels
    are not yet set.
    """
    if item.get("gold_auto") is True:
        title = (item.get("gold_document_title") or "").strip()
        if title:
            stmt = (
                select(Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.title.ilike(f"%{title}%"))
                .order_by(Chunk.chunk_index)
                .limit(1)
            )
            r = (await session.execute(stmt)).scalars().first()
            if r:
                return [r]
        # Demo fallback: first chunk in DB (ingest at least one document)
        stmt = (
            select(Chunk.id)
            .order_by(Chunk.document_id, Chunk.chunk_index)
            .limit(1)
        )
        r = (await session.execute(stmt)).scalars().first()
        return [r] if r else []

    raw = item.get("gold_chunk_ids")
    if isinstance(raw, list) and raw:
        out: list[UUID] = []
        for x in raw:
            try:
                out.append(UUID(str(x)) if not isinstance(x, UUID) else x)
            except Exception:
                logger.warning("invalid gold_chunk_id %r", x)
        return out

    title = (item.get("gold_document_title") or "").strip()
    idx = item.get("gold_chunk_index")
    if title and idx is not None:
        idx_i = int(idx)

        async def _one(stmt):
            r = (await session.execute(stmt)).scalars().first()
            return [r] if r else []

        # Exact title + chunk_index
        stmt = (
            select(Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.title == title, Chunk.chunk_index == idx_i)
            .limit(1)
        )
        out = await _one(stmt)
        if out:
            return out

        # Case-insensitive title (common mismatch)
        stmt = (
            select(Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                func.lower(func.trim(Document.title)) == title.lower(),
                Chunk.chunk_index == idx_i,
            )
            .limit(1)
        )
        out = await _one(stmt)
        if out:
            return out

        # Title contains (e.g. extra whitespace or subtitle in DB)
        stmt = (
            select(Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.title.ilike(f"%{title}%"),
                Chunk.chunk_index == idx_i,
            )
            .limit(1)
        )
        out = await _one(stmt)
        if out:
            return out

        logger.warning(
            "eval: no gold chunk for title=%r chunk_index=%s (check ingest titles)",
            title,
            idx_i,
        )

    return []


async def retrieval_previews_for_ids(
    session: AsyncSession,
    chunk_ids: list[UUID],
    *,
    excerpt_chars: int = 480,
) -> list[dict[str, str]]:
    """
    Ordered snippets for each chunk id (for Evaluation UI: basic vs agentic retrieved text).

    Returns ``[{chunk_id, document_title, chunk_index, excerpt}, ...]`` in the same order
    as ``chunk_ids`` (missing ids are skipped).
    """
    if not chunk_ids:
        return []
    stmt = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(chunk_ids))
    )
    rows = (await session.execute(stmt)).all()
    by_id: dict[UUID, tuple[Chunk, Document]] = {ch.id: (ch, doc) for ch, doc in rows}
    out: list[dict[str, str]] = []
    for cid in chunk_ids:
        pair = by_id.get(cid)
        if not pair:
            continue
        ch, doc = pair
        text = (ch.content or "").strip().replace("\n", " ")
        if len(text) > excerpt_chars:
            text = text[: excerpt_chars - 1] + "…"
        out.append(
            {
                "chunk_id": str(ch.id),
                "document_title": (doc.title or "")[:200],
                "chunk_index": str(int(ch.chunk_index)),
                "excerpt": text,
            }
        )
    return out


def recall_hit(gold: list[UUID], retrieved: list[UUID], *, k: int | None = None) -> bool:
    """True if any gold chunk id appears in the first ``k`` retrieved (or all retrieved if k is None)."""
    if not gold:
        return False
    take = retrieved if k is None else retrieved[:k]
    gset = set(gold)
    return any(rid in gset for rid in take)


def mean_metric(hits: list[bool]) -> float | None:
    if not hits:
        return None
    return sum(1 for h in hits if h) / len(hits)


def default_benchmark_path() -> Path:
    """``backend/eval/data/benchmark.json`` relative to CWD when running from ``backend/``."""
    return Path(__file__).resolve().parents[2] / "eval" / "data" / "benchmark.json"


def default_results_path() -> Path:
    return Path(__file__).resolve().parents[2] / "eval" / "results" / "latest.json"
