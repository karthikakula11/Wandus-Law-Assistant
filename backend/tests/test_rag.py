"""RAG tests with mocks."""

import os

import pytest
from sqlalchemy import delete

pytestmark = pytest.mark.integration


def _have_real_db() -> bool:
    u = os.environ.get("DATABASE_URL", "")
    return bool(u) and "59999" not in u


@pytest.mark.asyncio
async def test_chat_rag_uses_context(monkeypatch):
    if not _have_real_db():
        pytest.skip("Needs DATABASE_URL")

    from app.config import clear_settings_cache
    from app.database import dispose_db, get_session_factory
    from app.models import Chunk, Document
    from app.services import rag as rag_mod
    from app.services.ingest import ingest_document

    clear_settings_cache()
    await dispose_db()

    async def fake_embed(texts: list[str]):
        out = []
        for i, _ in enumerate(texts):
            v = [0.0] * 1536
            v[i % 1536] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("app.services.ingest.embed_texts", fake_embed)
    monkeypatch.setattr("app.services.rag.embed_texts", fake_embed)

    async def fake_generate_answer(messages):
        return "According to [1], the short title is Demo Act."

    monkeypatch.setattr(rag_mod, "generate_answer", fake_generate_answer)

    async with get_session_factory()() as session:
        doc_id, _ = await ingest_document(
            session,
            title="Demo Act",
            text="Section 1. This Act may be cited as the Demo Act. "
            "Section 2. The authority may issue rules." * 2,
            source_uri=None,
        )

    try:
        async with get_session_factory()() as session:
            answer, cites = await rag_mod.chat_rag(
                session, question="What is the short title?", top_k=3
            )
        assert "Demo" in answer or "short" in answer.lower()
        assert len(cites) >= 1
    finally:
        async with get_session_factory()() as session:
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))
            await session.execute(delete(Document).where(Document.id == doc_id))
            await session.commit()

    await dispose_db()
    clear_settings_cache()
