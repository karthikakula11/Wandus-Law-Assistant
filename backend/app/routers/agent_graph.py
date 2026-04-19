"""Expose LangGraph structure for UI (ASCII / optional Mermaid)."""

from fastapi import APIRouter

from app.services.rag_graph import export_rag_graph_ascii, export_rag_graph_mermaid

router = APIRouter(prefix="/rag-graph", tags=["rag-graph"])


@router.get("/ascii")
def get_rag_graph_ascii() -> dict[str, str]:
    """Return ASCII art from LangChain ``Graph.draw_ascii()`` for the compiled graph."""
    return {"ascii": export_rag_graph_ascii()}


@router.get("/mermaid")
def get_rag_graph_mermaid() -> dict[str, str]:
    """Return Mermaid for the **compiled** LangGraph (``__start__`` / ``__end__`` and all edges).

    Built from ``compiled.builder`` + LangChain ``Graph.draw_mermaid()`` — not a hand-maintained sketch.
    """
    return {"mermaid": export_rag_graph_mermaid()}
