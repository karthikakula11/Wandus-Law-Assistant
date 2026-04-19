"""List indexed documents and inspect chunk text for debugging RAG ingest."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Chunk, Document
from app.services.document_delete import delete_indexed_document
from app.schemas import (
    ChunkPreviewItem,
    DocumentChunksPreviewResponse,
    DocumentListItem,
    DocumentListResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])

_EXCERPT_MAX = 1200


@router.get("", response_model=DocumentListResponse)
async def list_documents(session: AsyncSession = Depends(get_session)):
    """Newest first; includes chunk counts for the UI."""
    stmt = (
        select(
            Document.id,
            Document.title,
            Document.created_at,
            func.count(Chunk.id).label("chunk_count"),
        )
        .select_from(Document)
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id, Document.title, Document.created_at)
        .order_by(Document.created_at.desc())
    )
    rows = (await session.execute(stmt)).all()
    documents = [
        DocumentListItem(
            id=row[0],
            title=row[1],
            created_at=row[2],
            chunk_count=int(row[3] or 0),
        )
        for row in rows
    ]
    return DocumentListResponse(documents=documents)


@router.delete("/{document_id}", status_code=204)
async def remove_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Unindex a document: remove OpenSearch entries for each chunk, then delete the
    document row (chunks removed by cascade). Frees database storage for embeddings.
    """
    ok = await delete_indexed_document(session, document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.commit()


@router.get("/{document_id}/chunks", response_model=DocumentChunksPreviewResponse)
async def get_document_chunks_preview(
    document_id: UUID,
    limit: int = Query(30, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """
    Return the start of each stored chunk (debug ingest / PDF extraction).
    If excerpts look like gibberish, re-upload after fixing the PDF or paste text instead.
    """
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    stmt = (
        select(Chunk.chunk_index, Chunk.content)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    chunks: list[ChunkPreviewItem] = []
    for chunk_index, content in rows:
        text = content or ""
        excerpt = text[:_EXCERPT_MAX] + ("…" if len(text) > _EXCERPT_MAX else "")
        chunks.append(
            ChunkPreviewItem(
                chunk_index=chunk_index,
                char_count=len(text),
                excerpt=excerpt,
            )
        )
    return DocumentChunksPreviewResponse(
        document_id=doc.id,
        title=doc.title,
        chunks=chunks,
    )
