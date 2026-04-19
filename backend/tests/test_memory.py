"""Long-term memory: normalization, prompt injection, API validation."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.models import Chunk, Document
from app.services.long_term_memory import normalize_user_key
from app.services.rag_prompts import RAGPromptBuilder


def test_normalize_user_key():
    assert normalize_user_key(None) is None
    assert normalize_user_key("") is None
    assert normalize_user_key("short") is None
    assert normalize_user_key("a" * 7) is None
    assert normalize_user_key("a" * 8) == "aaaaaaaa"
    assert normalize_user_key("  " + "b" * 10 + "  ") == "b" * 10
    assert normalize_user_key("bad!chars") is None


def test_rag_prompt_builder_includes_memory_block():
    doc_id = uuid.uuid4()
    ch_id = uuid.uuid4()
    doc = Document(id=doc_id, title="Act", source_uri=None)
    ch = Chunk(
        id=ch_id,
        document_id=doc_id,
        chunk_index=0,
        content="Section 1.",
        meta=None,
        embedding=[0.0] * 1536,
    )
    msgs, _ = RAGPromptBuilder().build_messages(
        "What is the short title?",
        [(ch, doc)],
        [],
        memory_snippets=["User prefers Indian law examples."],
    )
    assert msgs[-1]["role"] == "user"
    user = msgs[-1]["content"]
    assert "Long-term memory" in user
    assert "Indian law" in user
    assert "Section 1" in user


@pytest.mark.asyncio
async def test_memory_items_rejects_invalid_user_id():
    from app.config import clear_settings_cache
    from app.main import create_app

    clear_settings_cache()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/memory/items?memory_user_id=xx")
        assert r.status_code in (400, 422)
    clear_settings_cache()
