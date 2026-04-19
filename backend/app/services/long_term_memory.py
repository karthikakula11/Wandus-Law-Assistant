"""Semantic retrieval and writes for long-term chat memory (per ``user_key``)."""

from __future__ import annotations

import re
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import embed_memory_texts
from app.models import MemoryItem

MEMORY_TOP_K = 8
MAX_MEMORY_CHARS = 4000
# Opaque id from client (e.g. UUID); alphanumeric + underscore/hyphen.
_USER_KEY_RE = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")


def normalize_user_key(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s or not _USER_KEY_RE.match(s):
        return None
    return s


async def retrieve_memory_texts(
    session: AsyncSession,
    user_key: str,
    query: str,
    *,
    top_k: int = MEMORY_TOP_K,
) -> list[str]:
    """Return up to ``top_k`` memory strings most similar to ``query`` (cosine distance)."""
    (qvec,) = await embed_memory_texts([query])
    dist_col = MemoryItem.embedding.cosine_distance(qvec)
    stmt = (
        select(MemoryItem.content, dist_col.label("dist"))
        .where(MemoryItem.user_key == user_key)
        .order_by(dist_col)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    out: list[str] = []
    for content, _dist in rows:
        t = (content or "").strip()
        if t and t not in out:
            out.append(t)
    return out


async def add_memory_item(session: AsyncSession, user_key: str, content: str) -> MemoryItem:
    content = (content or "").strip()
    if not content:
        raise ValueError("content is empty")
    if len(content) > MAX_MEMORY_CHARS:
        content = content[:MAX_MEMORY_CHARS]
    (vec,) = await embed_memory_texts([content])
    row = MemoryItem(user_key=user_key, content=content, embedding=vec)
    session.add(row)
    await session.flush()
    return row


async def list_memory_items(
    session: AsyncSession, user_key: str, *, limit: int = 100
) -> list[MemoryItem]:
    stmt = (
        select(MemoryItem)
        .where(MemoryItem.user_key == user_key)
        .order_by(MemoryItem.created_at.desc())
        .limit(min(limit, 500))
    )
    return list((await session.execute(stmt)).scalars().all())


async def delete_memory_item(session: AsyncSession, user_key: str, item_id: UUID) -> bool:
    r = await session.execute(
        delete(MemoryItem).where(
            MemoryItem.id == item_id,
            MemoryItem.user_key == user_key,
        )
    )
    return (r.rowcount or 0) > 0


async def nearest_memory_cosine_distance(
    session: AsyncSession,
    user_key: str,
    text: str,
) -> float | None:
    """Smallest cosine distance from ``text``'s embedding to any row for ``user_key`` (None if empty)."""
    (qvec,) = await embed_memory_texts([text])
    dist_col = MemoryItem.embedding.cosine_distance(qvec)
    stmt = (
        select(dist_col)
        .where(MemoryItem.user_key == user_key)
        .order_by(dist_col.asc())
        .limit(1)
    )
    r = (await session.execute(stmt)).scalar_one_or_none()
    return float(r) if r is not None else None
