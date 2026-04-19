"""
Jam with AI /ask behavior: use top-k retrieval when chunks exist (no distance fallback unless gate on).
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_chat_auto_uses_rag_when_best_dense_is_poor_and_gate_disabled(monkeypatch):
    from app.services import rag as rag_mod

    async def fake_count(_session):
        return 5

    async def fake_retrieve(*_a, **_k):
        ch = MagicMock()
        ch.id = uuid4()
        ch.content = "Section 1. Short title: Demo Act."
        ch.chunk_index = 0
        doc = MagicMock()
        doc.id = uuid4()
        doc.title = "Test Act"
        return [(ch, doc, 0.99)], 0.99

    async def fake_gen(_messages):
        return "answer-from-rag"

    def gs():
        m = MagicMock()
        m.rag_distance_gate_enabled = False
        m.rag_distance_threshold = 0.85
        m.use_langgraph_agent = False
        return m

    monkeypatch.setattr(rag_mod, "get_settings", gs)
    monkeypatch.setattr(rag_mod, "count_chunks", fake_count)
    monkeypatch.setattr(rag_mod, "retrieve_for_chat", fake_retrieve)
    monkeypatch.setattr(rag_mod, "generate_answer", fake_gen)

    session = MagicMock()
    answer, cites, src = await rag_mod.chat_auto(
        session,
        "What is the short title of this Act?",
        top_k=5,
        history=None,
        scope_document_ids=None,
    )
    assert src == "documents"
    assert answer == "answer-from-rag"
    assert len(cites) >= 1


@pytest.mark.asyncio
async def test_chat_auto_falls_back_general_when_gate_on_and_dense_poor(monkeypatch):
    from app.services import rag as rag_mod

    async def fake_count(_session):
        return 5

    async def fake_retrieve(*_a, **_k):
        ch = MagicMock()
        ch.id = uuid4()
        ch.content = "Section 1. Short title: Demo Act."
        ch.chunk_index = 0
        doc = MagicMock()
        doc.id = uuid4()
        doc.title = "Test Act"
        return [(ch, doc, 0.99)], 0.99

    async def fake_general(_q, _h=None, **_kwargs):
        return "general-only"

    async def fake_gen(_messages):
        pytest.fail("generate_answer should not run when gate sends general")

    def gs():
        m = MagicMock()
        m.rag_distance_gate_enabled = True
        m.rag_distance_threshold = 0.85
        m.use_langgraph_agent = False
        return m

    monkeypatch.setattr(rag_mod, "get_settings", gs)
    monkeypatch.setattr(rag_mod, "count_chunks", fake_count)
    monkeypatch.setattr(rag_mod, "retrieve_for_chat", fake_retrieve)
    monkeypatch.setattr(rag_mod, "chat_general", fake_general)
    monkeypatch.setattr(rag_mod, "generate_answer", fake_gen)

    session = MagicMock()
    answer, cites, src = await rag_mod.chat_auto(
        session,
        "Obscure query unlikely to match embeddings",
        top_k=5,
        history=None,
        scope_document_ids=None,
    )
    assert src == "general"
    assert answer == "general-only"
    assert cites == []


def test_should_use_general_respects_gate(monkeypatch):
    from unittest.mock import MagicMock

    from app.services import rag as rag_mod

    def gs_off():
        m = MagicMock()
        m.rag_distance_gate_enabled = False
        m.rag_distance_threshold = 0.85
        return m

    def gs_on():
        m = MagicMock()
        m.rag_distance_gate_enabled = True
        m.rag_distance_threshold = 0.85
        return m

    monkeypatch.setattr(rag_mod, "get_settings", gs_off)
    assert rag_mod.should_use_general_instead_of_rag(0.99) is False

    monkeypatch.setattr(rag_mod, "get_settings", gs_on)
    assert rag_mod.should_use_general_instead_of_rag(0.99) is True
    assert rag_mod.should_use_general_instead_of_rag(0.1) is False
