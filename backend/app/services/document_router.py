"""
Scalable multi-document routing without listing every file to an LLM.

When several uploads exist and the user did not name a specific file, we take a
**broad dense pool** of nearest chunks (ANN), collect **distinct document_ids**
in relevance order, cap at **document_router_max_candidate_docs**, and run
hybrid retrieval only within those documents.

This scales to hundreds or thousands of indexed documents: cost is O(pool size),
not O(number of documents).
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Document
from app.services.document_scope import resolve_scope_document_id
from app.services.retrieval_hybrid import retrieve_dense_topn

logger = logging.getLogger(__name__)


async def count_documents(session: AsyncSession) -> int:
    n = await session.scalar(select(func.count()).select_from(Document))
    return int(n or 0)


async def shortlist_candidate_document_ids(
    session: AsyncSession,
    question: str,
) -> list[UUID] | None:
    """
    Distinct document IDs appearing in the top-N nearest chunks (global ANN pool).
    Returns None if the pool is empty; callers treat None as unrestricted search.
    """
    settings = get_settings()
    pool = settings.document_router_pool_chunks
    cap = settings.document_router_max_candidate_docs

    rows, _ = await retrieve_dense_topn(
        session,
        question,
        pool,
        scope_document_ids=None,
    )
    seen: set[UUID] = set()
    ordered: list[UUID] = []
    for ch, doc, _dist in rows:
        did = doc.id
        if did not in seen:
            seen.add(did)
            ordered.append(did)
            if len(ordered) >= cap:
                break

    if not ordered:
        return None

    logger.info(
        "phase=document_shortlist pool=%s distinct_docs=%s cap=%s",
        pool,
        len(ordered),
        cap,
    )
    return ordered


async def prepare_retrieval_scope(
    session: AsyncSession,
    question: str,
    history: list[dict] | None,
    explicit_document_id: UUID | None,
) -> list[UUID] | None:
    """
    Final scope for chunk retrieval:

    - ``[uuid]`` — user named a file / explicit client id / history match.
    - ``[...many...]`` — embedding shortlist over many uploads (multi-doc library).
    - ``None`` — single-doc library or router disabled: search all chunks (unscoped).
    """
    named = await resolve_scope_document_id(
        session,
        question,
        history,
        explicit_document_id,
    )
    if named is not None:
        logger.info("phase=retrieval_scope source=named id=%s", named)
        return [named]

    n_docs = await count_documents(session)
    if n_docs <= 1:
        return None

    settings = get_settings()
    if not settings.document_router_enabled:
        return None

    return await shortlist_candidate_document_ids(session, question)
