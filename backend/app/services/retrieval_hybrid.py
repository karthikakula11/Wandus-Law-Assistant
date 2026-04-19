"""
Hybrid retrieval: dense cosine (pgvector) + BM25 (OpenSearch), fused with RRF.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.embeddings import embed_texts
from app.models import Chunk, Document
from app.services import search_index
from app.services.rrf_util import rrf_fuse

logger = logging.getLogger(__name__)


def _apply_scope_ids(stmt, scope_document_ids: list[UUID] | None):
    if scope_document_ids:
        return stmt.where(Chunk.document_id.in_(scope_document_ids))
    return stmt


async def retrieve_dense_topn(
    session: AsyncSession,
    question: str,
    limit: int,
    *,
    scope_document_ids: list[UUID] | None = None,
) -> tuple[list[tuple[Chunk, Document, float]], float]:
    """
    Cosine distance ascending. Returns rows and best (minimum) distance in the list.
    """
    (qvec,) = await embed_texts([question])
    dist_col = Chunk.embedding.cosine_distance(qvec)
    stmt = (
        select(Chunk, Document, dist_col.label("dist"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(dist_col)
        .limit(limit)
    )
    stmt = _apply_scope_ids(stmt, scope_document_ids)
    rows = (await session.execute(stmt)).all()
    out: list[tuple[Chunk, Document, float]] = []
    best = 1.0
    for chunk, doc, dist in rows:
        d = float(dist)
        out.append((chunk, doc, d))
        best = min(best, d)
    if not out:
        best = 1.0
    return out, best


async def retrieve_bm25_topn(
    question: str,
    limit: int,
    *,
    scope_document_ids: list[UUID] | None = None,
) -> list[UUID]:
    pairs = await search_index.search_bm25(
        question, limit, scope_document_ids=scope_document_ids
    )
    return [cid for cid, _ in pairs]


async def _load_chunks_map(
    session: AsyncSession,
    ids: list[UUID],
) -> dict[UUID, tuple[Chunk, Document]]:
    if not ids:
        return {}
    stmt = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(ids))
    )
    rows = (await session.execute(stmt)).all()
    return {ch.id: (ch, doc) for ch, doc in rows}


async def retrieve_hybrid_rrf(
    session: AsyncSession,
    question: str,
    top_k: int,
    *,
    per_list_cap: int | None = None,
    scope_document_ids: list[UUID] | None = None,
) -> tuple[list[tuple[Chunk, Document, float]], float]:
    """
    RRF fusion of dense + BM25 lists; third float is fused RRF score (larger = better).
    `best_dense_dist` is the best cosine distance from the dense leg (for gating).

    When ``OPENSEARCH_UNIFIED_HYBRID`` is true, uses Jam-style **single** OpenSearch hybrid + RRF
    (see ``opensearch_unified.search_unified_native``) instead of PG dense + BM25 + Python RRF.
    """
    settings = get_settings()
    cap = per_list_cap if per_list_cap is not None else settings.hybrid_per_list_cap

    if settings.opensearch_unified_hybrid and (settings.opensearch_url or "").strip():
        return await _retrieve_opensearch_unified_jam(
            session,
            question,
            top_k,
            per_list_cap=cap,
            scope_document_ids=scope_document_ids,
        )

    rrf_k = settings.rrf_k

    dense_rows, best_dense_dist = await retrieve_dense_topn(
        session, question, cap, scope_document_ids=scope_document_ids
    )
    dense_ids = [ch.id for ch, _, _ in dense_rows]

    bm25_ids: list[UUID] = []
    if (
        settings.hybrid_rag_enabled
        and settings.bm25_enabled
        and (settings.opensearch_url or "").strip()
    ):
        try:
            bm25_ids = await retrieve_bm25_topn(
                question, cap, scope_document_ids=scope_document_ids
            )
        except Exception as e:
            logger.warning("BM25 leg failed, dense only: %s", e)
            bm25_ids = []

    fused = rrf_fuse(dense_ids, bm25_ids, k=rrf_k)
    if not fused:
        return [], best_dense_dist

    ordered_ids = sorted(fused.keys(), key=lambda i: -fused[i])
    ordered_ids = ordered_ids[:top_k]

    loaded = await _load_chunks_map(session, ordered_ids)
    out: list[tuple[Chunk, Document, float]] = []
    for cid in ordered_ids:
        pair = loaded.get(cid)
        if pair:
            ch, doc = pair
            out.append((ch, doc, fused[cid]))
    return out, best_dense_dist


async def _retrieve_opensearch_unified_jam(
    session: AsyncSession,
    question: str,
    top_k: int,
    *,
    per_list_cap: int,
    scope_document_ids: list[UUID] | None = None,
) -> tuple[list[tuple[Chunk, Document, float]], float]:
    """
    Jam course ``search_unified``: one hybrid query + RRF pipeline in OpenSearch, then load PG rows.
    """
    from app.services.opensearch_unified import search_unified_native

    vecs = await embed_texts([question])
    qvec = vecs[0] if vecs else []
    if not qvec:
        return [], 1.0

    pairs = await search_unified_native(
        question,
        qvec,
        min(top_k, per_list_cap),
        scope_document_ids=scope_document_ids,
    )
    dense_rows, best_dense_dist = await retrieve_dense_topn(
        session, question, per_list_cap, scope_document_ids=scope_document_ids
    )
    if not pairs:
        return [], best_dense_dist

    ordered_ids = [cid for cid, _ in pairs][:top_k]
    score_by_id = {cid: sc for cid, sc in pairs}
    loaded = await _load_chunks_map(session, ordered_ids)
    out: list[tuple[Chunk, Document, float]] = []
    for cid in ordered_ids:
        pair = loaded.get(cid)
        if pair:
            ch, doc = pair
            out.append((ch, doc, float(score_by_id.get(cid, 0.0))))
    return out, best_dense_dist


async def retrieve_for_single_query(
    session: AsyncSession,
    question: str,
    take_k: int,
    *,
    per_list_cap: int | None = None,
    scope_document_ids: list[UUID] | None = None,
) -> tuple[list[tuple[Chunk, Document, float]], float]:
    """
    Hybrid RRF when `hybrid_rag_enabled`; else dense-only with score = -distance for merge.
    """
    settings = get_settings()
    cap = per_list_cap if per_list_cap is not None else settings.hybrid_per_list_cap
    q = question.strip()
    if not q:
        return [], 1.0

    if settings.hybrid_rag_enabled:
        return await retrieve_hybrid_rrf(
            session, q, take_k, per_list_cap=cap, scope_document_ids=scope_document_ids
        )

    rows, bd = await retrieve_dense_topn(session, q, cap, scope_document_ids=scope_document_ids)
    rows = rows[:take_k]
    scored = [(ch, doc, -d) for ch, doc, d in rows]
    return scored, bd


async def retrieve_merged_queries(
    session: AsyncSession,
    queries: list[str],
    top_k: int,
    *,
    per_list_cap: int | None = None,
    scope_document_ids: list[UUID] | None = None,
) -> tuple[list[tuple[Chunk, Document, float]], float]:
    """Run retrieval per query; merge by chunk id keeping best score; global min dense distance."""
    if not queries:
        return [], 1.0
    settings = get_settings()
    cap = per_list_cap if per_list_cap is not None else settings.hybrid_per_list_cap

    merged: dict[UUID, tuple[Chunk, Document, float]] = {}
    best_dense_global = 1.0

    for q in queries:
        rows, bd = await retrieve_for_single_query(
            session,
            q.strip(),
            cap,
            per_list_cap=cap,
            scope_document_ids=scope_document_ids,
        )
        best_dense_global = min(best_dense_global, bd)
        for ch, doc, score in rows:
            prev = merged.get(ch.id)
            if prev is None or score > prev[2]:
                merged[ch.id] = (ch, doc, score)

    sorted_rows = sorted(merged.values(), key=lambda x: -x[2])
    return sorted_rows[:top_k], best_dense_global
