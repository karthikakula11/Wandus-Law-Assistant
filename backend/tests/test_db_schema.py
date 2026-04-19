"""Integration: requires Postgres from docker compose and DATABASE_URL."""

import os
import uuid

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.integration


def _have_real_db() -> bool:
    u = os.environ.get("DATABASE_URL", "")
    return bool(u) and "59999" not in u


@pytest.mark.asyncio
async def test_document_chunk_fk_and_vector():
    if not _have_real_db():
        pytest.skip("Set DATABASE_URL to running Postgres")

    from app.config import clear_settings_cache
    from app.database import dispose_db, get_session_factory
    from app.models import Chunk, Document, EMBED_DIM

    clear_settings_cache()
    await dispose_db()

    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    zero = [0.0] * EMBED_DIM

    async with get_session_factory()() as session:
        session.add(
            Document(id=doc_id, title="Test Act", source_uri=None)
        )
        session.add(
            Chunk(
                id=chunk_id,
                document_id=doc_id,
                chunk_index=0,
                content="Section 1. Test.",
                meta={"section": "1"},
                embedding=zero,
            )
        )
        await session.commit()

    async with get_session_factory()() as session:
        ch = await session.scalar(select(Chunk).where(Chunk.id == chunk_id))
        assert ch is not None
        assert ch.document_id == doc_id
        assert len(ch.embedding) == EMBED_DIM

        await session.delete(ch)
        doc = await session.scalar(select(Document).where(Document.id == doc_id))
        await session.delete(doc)
        await session.commit()

    await dispose_db()
    clear_settings_cache()
