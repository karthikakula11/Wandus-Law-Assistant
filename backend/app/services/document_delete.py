"""Remove an indexed document: OpenSearch rows first, then Postgres (chunks cascade)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.services.search_index import delete_chunk_document


async def delete_indexed_document(session: AsyncSession, document_id: UUID) -> bool:
    """
    Return False if the document does not exist.
    Caller should ``commit`` the session on success.
    """
    doc = await session.get(Document, document_id)
    if doc is None:
        return False

    r = await session.execute(select(Chunk.id).where(Chunk.document_id == document_id))
    chunk_ids: list[UUID] = list(r.scalars().all())
    for cid in chunk_ids:
        await delete_chunk_document(cid)

    await session.delete(doc)
    return True
