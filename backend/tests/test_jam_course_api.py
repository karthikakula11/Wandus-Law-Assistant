"""Jam with AI–compatible /api/v1/ask flow (no OpenAI / DB when mocked)."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.jam_schemas import AskRequest
from app.services.jam_ask_service import run_jam_ask


@pytest.mark.asyncio
async def test_jam_ask_empty_index_message(monkeypatch):
    async def empty_retrieve(*_a, **_k):
        return [], 1.0

    monkeypatch.setattr(
        "app.services.jam_ask_service.retrieve_for_chat",
        empty_retrieve,
    )
    session = MagicMock()
    req = AskRequest(query="test", top_k=3, use_hybrid=True, model="gpt-4o-mini")
    out = await run_jam_ask(session, req)
    assert out.chunks_used == 0
    assert "couldn't find" in out.answer.lower()
    assert out.search_mode == "hybrid"


@pytest.mark.asyncio
async def test_jam_ask_builds_prompt_and_calls_openai(monkeypatch):
    ch = MagicMock()
    ch.id = uuid4()
    ch.content = "Section 1. Short title: Demo Act."
    doc = MagicMock()
    doc.id = uuid4()
    doc.title = "Demo Act"
    doc.source_uri = None

    async def fake_retrieve(*_a, **_k):
        return [(ch, doc, 0.1)], 0.1

    async def fake_gen(prompt: str, model: str) -> str:
        assert "### Context from Papers:" in prompt
        assert "### Question:" in prompt
        assert "Demo Act" in prompt
        return "The short title is Demo Act."

    monkeypatch.setattr(
        "app.services.jam_ask_service.retrieve_for_chat",
        fake_retrieve,
    )
    monkeypatch.setattr(
        "app.services.jam_ask_service.generate_jam_openai",
        fake_gen,
    )

    session = MagicMock()
    req = AskRequest(query="What is the short title?", top_k=3, use_hybrid=False)
    out = await run_jam_ask(session, req)
    assert out.answer == "The short title is Demo Act."
    assert out.chunks_used == 1
    assert out.search_mode == "bm25"
