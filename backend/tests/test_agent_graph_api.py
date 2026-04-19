import pytest


@pytest.mark.asyncio
async def test_rag_graph_ascii_returns_diagram(client):
    r = await client.get("/rag-graph/ascii")
    assert r.status_code == 200
    body = r.json()
    assert "ascii" in body
    text = body["ascii"]
    assert len(text) > 20
    assert "plan" in text
    assert "grade" in text
    # Box-drawing style from LangChain draw_ascii
    assert "+" in text or "-" in text or "|" in text


@pytest.mark.asyncio
async def test_rag_graph_mermaid_returns_diagram(client):
    r = await client.get("/rag-graph/mermaid")
    assert r.status_code == 200
    body = r.json()
    assert "mermaid" in body
    text = body["mermaid"]
    assert len(text) > 30
    # Compiled LangGraph topology: START/END + nodes + conditional edges from grade
    assert "__start__" in text
    assert "__end__" in text
    assert "plan" in text
    assert "retrieve" in text
    assert "grade" in text
    assert "rewrite" in text
    assert "broaden" in text
    assert "generate" in text
    assert "graph td" in text.lower()
    assert ".->" in text  # conditional (dotted) edges from grade
