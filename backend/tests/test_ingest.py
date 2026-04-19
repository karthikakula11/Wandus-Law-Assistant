"""Ingest with mocked embeddings."""

import os

import pytest

pytestmark = pytest.mark.integration


def _have_real_db() -> bool:
    u = os.environ.get("DATABASE_URL", "")
    return bool(u) and "59999" not in u


@pytest.mark.asyncio
async def test_ingest_document_mock_embeddings(monkeypatch):
    if not _have_real_db():
        pytest.skip("Needs DATABASE_URL")

    from app.config import clear_settings_cache
    from app.database import dispose_db, get_session_factory
    from app.models import Chunk, Document
    from app.services import ingest as ingest_mod
    from sqlalchemy import delete, func, select

    clear_settings_cache()
    await dispose_db()

    async def fake_embed(texts: list[str]):
        return [[0.01 * (i % 7)] * 1536 for i in range(len(texts))]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)

    text = (
        "Section 1. Short title. This Act may be cited as the Example Act.\n\n"
        "Section 2. Definitions. In this Act, unless the context otherwise requires, "
        '"Authority" means the regulatory authority established under section 5.'
    ) * 3

    async with get_session_factory()() as session:
        doc_id, n = await ingest_mod.ingest_document(
            session,
            title="Example Act",
            text=text,
            source_uri="file:///example.txt",
        )

    assert n >= 1

    async with get_session_factory()() as session:
        cnt = await session.scalar(select(func.count()).select_from(Chunk))
        assert cnt is not None and cnt >= n

        await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
        await session.execute(delete(Document).where(Document.id == doc_id))
        await session.commit()

    await dispose_db()
    clear_settings_cache()
