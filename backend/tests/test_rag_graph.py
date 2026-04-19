"""LangGraph path with mocks (no OpenAI)."""

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_run_agentic_rag_empty_retrieval_uses_general(monkeypatch):
    from app.services import rag_graph as rg

    async def fake_plan(q: str):
        return ["q1"]

    async def fake_merged(*args, **kwargs):
        return [], 1.0

    async def fake_chat(question: str, history=None, **kwargs):
        return "general-answer"

    def fake_settings():
        return SimpleNamespace(
            hybrid_per_list_cap=50,
            rag_agent_max_rewrite_rounds=0,
            rag_agentic_grade_enabled=False,
            rag_agent_skip_plan_llm=False,
            rag_distance_gate_enabled=False,
        )

    monkeypatch.setattr(rg, "_plan_queries", fake_plan)
    monkeypatch.setattr(rg, "retrieve_merged_queries", fake_merged)
    monkeypatch.setattr(rg, "chat_general", fake_chat)
    monkeypatch.setattr(rg, "get_settings", fake_settings)
    monkeypatch.setattr(rg, "_graph", None)

    class _Sess:
        pass

    answer, cites, src = await rg.run_agentic_rag(_Sess(), "test question", 5, [])
    assert answer == "general-answer"
    assert cites == []
    assert src == "general"


def test_rag_graph_compiles():
    from app.services.rag_graph import build_rag_graph

    g = build_rag_graph()
    assert g is not None


def test_route_after_grade_empty_rows_rewrites_when_allowed(monkeypatch):
    from app.services import rag_graph as rg

    monkeypatch.setattr(
        rg,
        "get_settings",
        lambda: SimpleNamespace(
            rag_agent_max_rewrite_rounds=2,
            rag_distance_gate_enabled=False,
        ),
    )
    assert rg._route_after_grade({"rows": [], "rewrite_count": 0}) == "rewrite"
    assert rg._route_after_grade({"rows": [], "rewrite_count": 2}) == "generate"


def test_route_after_grade_relevant_then_broaden_when_gate_on(monkeypatch):
    from app.services import rag_graph as rg

    monkeypatch.setattr(
        rg,
        "get_settings",
        lambda: SimpleNamespace(
            rag_agent_max_rewrite_rounds=2,
            rag_distance_gate_enabled=True,
        ),
    )
    monkeypatch.setattr(
        rg,
        "should_use_general_instead_of_rag",
        lambda _bd: True,
    )
    fake_row = object()
    # rows present, relevant, distance gate says "weak" → broaden once
    assert (
        rg._route_after_grade(
            {
                "rows": [(fake_row, fake_row, 0.0)],
                "grade_relevant": True,
                "best_dense_dist": 0.9,
                "did_broaden": False,
            }
        )
        == "broaden"
    )


@pytest.mark.asyncio
async def test_stream_agentic_rag_events_emits_graph_steps(monkeypatch):
    """SSE-style stream includes graph_step before meta/token/done."""
    from app.services import rag_graph as rg

    async def fake_plan(q: str):
        return ["q1"]

    async def fake_merged(*args, **kwargs):
        return [], 1.0

    async def fake_chat(question: str, history=None, **kwargs):
        return "general-answer"

    def fake_settings():
        return SimpleNamespace(
            hybrid_per_list_cap=50,
            rag_agent_max_rewrite_rounds=0,
            rag_agentic_grade_enabled=False,
            rag_agent_skip_plan_llm=False,
            rag_distance_gate_enabled=False,
        )

    monkeypatch.setattr(rg, "_plan_queries", fake_plan)
    monkeypatch.setattr(rg, "retrieve_merged_queries", fake_merged)
    monkeypatch.setattr(rg, "chat_general", fake_chat)
    monkeypatch.setattr(rg, "get_settings", fake_settings)
    monkeypatch.setattr(rg, "_graph", None)

    class _Sess:
        pass

    out: list = []
    async for ev in rg.stream_agentic_rag_events(_Sess(), "q", 5, []):
        out.append(ev)

    steps = [e["node"] for e in out if e.get("event") == "graph_step"]
    assert "plan" in steps
    assert "retrieve" in steps
    assert "generate" in steps
    assert out[-1] == {"event": "done"}
    assert out[-2]["event"] == "token"
    assert out[-3]["event"] == "meta"
