"""
LangGraph agent: plan → retrieve → grade (Jam-style) → rewrite loop →
optional broaden → generate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Graph as LCGraph
from langgraph.graph import END, START, StateGraph
from app.services.openai_factory import get_async_openai_client
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.config import get_settings
from app.monitoring.openai_usage import record_chat_completion_usage
from app.schemas import CitationOut
from app.services.drift_detection import record_retrieval_confidence_sample
from app.services.rag import (
    ReplySource,
    build_rag_messages,
    chat_general,
    citations_from_contexts,
    generate_answer,
    sanitize_history,
    should_use_general_instead_of_rag,
)
from app.services.langfuse_tracing import build_langgraph_run_config, openai_trace_kwargs
from app.services.retrieval_hybrid import retrieve_merged_queries

logger = logging.getLogger(__name__)

PLAN_SYSTEM = (
    "You output only valid JSON. Given a legal study question, propose up to 2 short "
    'search queries for a document index. Schema: {"queries": ["...", "..."]}'
)

GRADE_SYSTEM = (
    "You evaluate whether retrieved legal document excerpts are sufficient and on-topic "
    'to answer the user\'s question. Reply with JSON only: '
    '{"binary_score": "yes" | "no", "reasoning": "brief"}. '
    'Use "yes" if excerpts contain relevant material that could ground a solid answer. '
    'Use "no" if excerpts are empty, clearly off-topic, or obviously insufficient.'
)

REWRITE_SYSTEM = (
    "You output only valid JSON. Prior retrieval was weak or empty. Propose 1–2 "
    "alternative short search queries for a legal document index (different wording "
    'or focus). Schema: {"queries": ["...", "..."]}'
)


class _GradeJson(BaseModel):
    binary_score: Literal["yes", "no"]
    reasoning: str = ""


async def _plan_queries(question: str) -> list[str]:
    settings = get_settings()
    client = get_async_openai_client()
    try:
        resp = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
            max_tokens=256,
            response_format={"type": "json_object"},
            **openai_trace_kwargs(name="agent-plan-queries"),
        )
        await record_chat_completion_usage(resp, route="agent-plan-queries")
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        qs = data.get("queries") or []
        out = [str(q).strip() for q in qs if str(q).strip()][:2]
        return out if out else [question]
    except Exception as e:
        logger.warning("plan_queries failed: %s", e)
        return [question]


def _context_for_grade(
    rows: list[tuple[Any, Any, float]],
    *,
    max_chars: int = 12000,
) -> str:
    parts: list[str] = []
    n = 0
    for ch, doc, _dist in rows:
        n += 1
        text = (ch.content or "").strip()
        if len(text) > 4000:
            text = text[:4000] + "…"
        parts.append(f"[{n}] doc={doc.title!r}\n{text}")
        if sum(len(p) for p in parts) >= max_chars:
            break
    return "\n\n---\n\n".join(parts)


async def _grade_documents_llm(question: str, context: str) -> tuple[bool, str]:
    settings = get_settings()
    client = get_async_openai_client()
    user = f"Question:\n{question}\n\nRetrieved excerpts:\n{context}"
    try:
        resp = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": GRADE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
            **openai_trace_kwargs(name="agent-grade-documents"),
        )
        await record_chat_completion_usage(resp, route="agent-grade-documents")
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _GradeJson.model_validate_json(raw)
        ok = parsed.binary_score == "yes"
        return ok, parsed.reasoning
    except Exception as e:
        logger.warning("grade_documents_llm failed: %s", e)
        # Prefer answering over spinning rewrite loops on API errors.
        heur = len(context.strip()) > 80
        return heur, f"fallback: {e!s}"


async def _rewrite_queries_llm(
    question: str,
    previous_queries: list[str],
    attempt: int,
) -> list[str]:
    settings = get_settings()
    client = get_async_openai_client()
    user = (
        f"Question: {question}\n"
        f"Previous queries tried: {previous_queries}\n"
        f"Rewrite round: {attempt}"
    )
    try:
        resp = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            **openai_trace_kwargs(name="agent-rewrite-queries"),
        )
        await record_chat_completion_usage(resp, route="agent-rewrite-queries")
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
        qs = data.get("queries") or []
        out = [str(q).strip() for q in qs if str(q).strip()][:2]
        return out if out else [question]
    except Exception as e:
        logger.warning("rewrite_queries_llm failed: %s", e)
        return [question]


async def _broaden_keywords(question: str) -> str:
    settings = get_settings()
    client = get_async_openai_client()
    prompt = (
        "Reply with a single line of 3–8 keywords (no JSON) to search a law library "
        f"for this question: {question!r}"
    )
    resp = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        **openai_trace_kwargs(name="agent-broaden-keywords"),
    )
    await record_chat_completion_usage(resp, route="agent-broaden-keywords")
    line = (resp.choices[0].message.content or "").strip().split("\n")[0]
    return line or question


async def _plan_node(state: dict[str, Any]) -> dict[str, Any]:
    q = state["question"]
    settings = get_settings()
    if settings.rag_agent_skip_plan_llm:
        queries = [q.strip()] if (q or "").strip() else [q]
        logger.info("phase=langgraph_plan skip_llm queries=%s", queries)
        return {**state, "search_queries": queries}
    queries = await _plan_queries(q)
    logger.info(
        "phase=langgraph_plan queries=%s",
        queries,
    )
    return {**state, "search_queries": queries}


async def _retrieve_node(
    state: dict[str, Any], config: RunnableConfig
) -> dict[str, Any]:
    configurable = config.get("configurable") or {}
    session: AsyncSession = configurable["session"]
    top_k = int(configurable.get("top_k", 5))
    per_cap = configurable.get("per_list_cap")

    queries = state.get("search_queries") or [state["question"]]
    scope_docs = configurable.get("scope_document_ids")
    rows, best_dense = await retrieve_merged_queries(
        session,
        queries,
        top_k,
        per_list_cap=per_cap,
        scope_document_ids=scope_docs,
    )
    logger.info(
        "phase=langgraph_retrieve queries=%s chunk_count=%s best_dense=%s",
        queries,
        len(rows),
        best_dense,
    )
    return {**state, "rows": rows, "best_dense_dist": best_dense}


async def _grade_node(state: dict[str, Any]) -> dict[str, Any]:
    rows = state.get("rows") or []
    settings = get_settings()
    if not rows:
        logger.info("phase=langgraph_grade no_chunks → route rewrite or generate")
        return {**state, "grade_relevant": False}

    if not settings.rag_agentic_grade_enabled:
        return {**state, "grade_relevant": True}

    ctx = _context_for_grade(rows)
    ok, reason = await _grade_documents_llm(state["question"], ctx)
    logger.info(
        "phase=langgraph_grade relevant=%s reason=%s",
        ok,
        reason[:200] if reason else "",
    )
    return {**state, "grade_relevant": ok}


def _route_after_grade(state: dict[str, Any]) -> Literal["rewrite", "broaden", "generate"]:
    settings = get_settings()
    rows = state.get("rows") or []
    rw = int(state.get("rewrite_count", 0))
    max_rw = settings.rag_agent_max_rewrite_rounds

    if not rows:
        if rw < max_rw:
            return "rewrite"
        return "generate"

    if not state.get("grade_relevant", True):
        if rw < max_rw:
            return "rewrite"
        return "generate"

    if settings.rag_distance_gate_enabled:
        bd = float(state.get("best_dense_dist", 1.0))
        if should_use_general_instead_of_rag(bd) and not state.get("did_broaden"):
            return "broaden"
    return "generate"


async def _rewrite_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    rw = int(state.get("rewrite_count", 0)) + 1
    prev = list(state.get("search_queries") or [question])
    new_qs = await _rewrite_queries_llm(question, prev, rw)
    logger.info("phase=langgraph_rewrite round=%s queries=%s", rw, new_qs)
    return {**state, "search_queries": new_qs, "rewrite_count": rw}


async def _broaden_node(state: dict[str, Any]) -> dict[str, Any]:
    q = state["question"]
    extra = await _broaden_keywords(q)
    prev = list(state.get("search_queries") or [])
    merged = prev + [extra]
    logger.info("phase=langgraph_broaden extra_query=%s", extra)
    return {**state, "search_queries": merged, "did_broaden": True}


async def _generate_node(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    history = sanitize_history(state.get("history"))
    rows = state.get("rows") or []

    memory_snippets = state.get("memory_snippets") or []

    if not rows:
        answer = await chat_general(question, history, memory_snippets=memory_snippets)
        return {
            **state,
            "answer": answer,
            "citations": [],
            "source": "general",
        }

    best_dense = float(state.get("best_dense_dist", 1.0))
    if should_use_general_instead_of_rag(best_dense):
        answer = await chat_general(question, history, memory_snippets=memory_snippets)
        return {
            **state,
            "answer": answer,
            "citations": [],
            "source": "general",
        }

    contexts = [(ch, doc) for ch, doc, _ in rows]
    messages, included = build_rag_messages(
        question, contexts, history, memory_snippets=memory_snippets
    )
    answer = await generate_answer(messages)
    cites = citations_from_contexts(included)
    logger.info(
        "phase=langgraph_generate citations=%s source=documents",
        len(cites),
    )
    return {
        **state,
        "answer": answer,
        "citations": cites,
        "source": "documents",
    }


def build_rag_graph() -> Any:
    g = StateGraph(dict)
    g.add_node("plan", _plan_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("grade", _grade_node)
    g.add_node("rewrite", _rewrite_node)
    g.add_node("broaden", _broaden_node)
    g.add_node("generate", _generate_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "rewrite": "rewrite",
            "broaden": "broaden",
            "generate": "generate",
        },
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("broaden", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


_graph = None


def get_rag_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


def compiled_state_graph_to_langchain_graph(compiled: Any) -> "LCGraph":
    """Rebuild a LangChain drawable ``Graph`` from the **compiled** LangGraph’s ``builder``.

    This is the **actual post-``compile()`` topology**: the same node ids, static edges, and
    conditional branches LangGraph recorded when you called ``add_edge`` / ``add_conditional_edges``
    (including ``__start__`` → first node and last node → ``__end__``).

    We cannot call ``compiled.get_graph().draw_mermaid()`` here: LangGraph’s ``get_graph()`` runs a
    Pregel simulation that applies channel writes and **raises** on plain ``dict`` state
    (``InvalidUpdateError`` on ``__root__``). Reading ``compiled.builder`` avoids that while staying
    faithful to the compiled graph.
    """
    from langchain_core.runnables.graph import Graph as LCGraph

    b = compiled.builder
    node_ids: set[str] = set(b.nodes.keys())
    for src, dst in b.edges:
        node_ids.add(src)
        node_ids.add(dst)
    for node_name, branch_map in b.branches.items():
        node_ids.add(node_name)
        for _path_name, bs in branch_map.items():
            for _label, target in bs.ends.items():
                node_ids.add(target)

    g = LCGraph()
    objs: dict[str, Any] = {}
    for nid in sorted(node_ids):
        objs[nid] = g.add_node(None, id=nid)

    for src, dst in sorted(b.edges, key=lambda t: (t[0], t[1])):
        g.add_edge(objs[src], objs[dst])

    for _src_name, branch_map in b.branches.items():
        for _path_name, bs in branch_map.items():
            for label, target in bs.ends.items():
                g.add_edge(objs[_src_name], objs[target], data=label, conditional=True)

    return g


def compiled_rag_graph_mermaid(compiled: Any) -> str:
    """Mermaid from LangChain's ``draw_mermaid`` — same family as LangGraph's compiled graph diagrams."""
    return compiled_state_graph_to_langchain_graph(compiled).draw_mermaid()


def export_rag_graph_mermaid() -> str:
    """Mermaid for the live compiled RAG graph — same structure LangGraph Studio would list (nodes + edges)."""
    return compiled_rag_graph_mermaid(get_rag_graph())


def compiled_rag_graph_ascii(compiled: Any) -> str:
    """ASCII art from LangChain's ``Graph.draw_ascii()`` (requires ``grandalf``)."""
    return compiled_state_graph_to_langchain_graph(compiled).draw_ascii()


def export_rag_graph_ascii() -> str:
    """ASCII diagram for the live compiled RAG graph (terminal-style boxes and edges)."""
    return compiled_rag_graph_ascii(get_rag_graph())


def _agentic_initial_state(
    question: str,
    history: list[dict] | None,
    memory_snippets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "history": history or [],
        "memory_snippets": memory_snippets or [],
        "did_broaden": False,
        "rewrite_count": 0,
    }


def _agentic_runnable_config(
    session: AsyncSession,
    top_k: int,
    *,
    scope_document_ids: list[UUID] | None,
) -> RunnableConfig:
    settings = get_settings()
    base: RunnableConfig = {
        "configurable": {
            "session": session,
            "top_k": top_k,
            "per_list_cap": settings.hybrid_per_list_cap,
            "scope_document_ids": scope_document_ids,
        }
    }
    return build_langgraph_run_config(
        base,
        trace_name="wandus-agentic-rag",
    )


async def accumulate_agentic_state_with_trace(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None,
    *,
    scope_document_ids: list[UUID] | None = None,
    memory_snippets: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Run the LangGraph RAG agent once; return final state and ordered node ids (for eval / demos)."""
    graph = get_rag_graph()
    initial = _agentic_initial_state(question, history, memory_snippets)
    cfg = _agentic_runnable_config(session, top_k, scope_document_ids=scope_document_ids)
    acc: dict[str, Any] = {**initial}
    trace: list[str] = []
    async for update in graph.astream(initial, cfg, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, payload in update.items():
            trace.append(node_name)
            acc.update(payload)
    return acc, trace


async def _accumulate_agentic_rag(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None,
    *,
    scope_document_ids: list[UUID] | None = None,
    memory_snippets: list[str] | None = None,
) -> dict[str, Any]:
    """Run the compiled graph once; merge each node update into accumulated state."""
    acc, _trace = await accumulate_agentic_state_with_trace(
        session,
        question,
        top_k,
        history,
        scope_document_ids=scope_document_ids,
        memory_snippets=memory_snippets,
    )
    return acc


async def run_agentic_rag(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None,
    *,
    scope_document_ids: list[UUID] | None = None,
    memory_snippets: list[str] | None = None,
) -> tuple[str, list[CitationOut], ReplySource]:
    acc = await _accumulate_agentic_rag(
        session,
        question,
        top_k,
        history,
        scope_document_ids=scope_document_ids,
        memory_snippets=memory_snippets,
    )
    src = acc.get("source") or "general"
    if src not in ("documents", "general"):
        src = "general"
    if src == "documents":
        bd = acc.get("best_dense_dist")
        if bd is not None:
            await record_retrieval_confidence_sample(session, float(bd))
    return (
        acc.get("answer") or "",
        acc.get("citations") or [],
        src,
    )


async def stream_agentic_rag_events(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None,
    *,
    scope_document_ids: list[UUID] | None = None,
    memory_snippets: list[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """SSE-friendly events: ``graph_step`` per node, then ``meta``, ``token``, ``done``."""
    graph = get_rag_graph()
    initial = _agentic_initial_state(question, history, memory_snippets)
    cfg = _agentic_runnable_config(session, top_k, scope_document_ids=scope_document_ids)
    acc: dict[str, Any] = {**initial}
    try:
        async for update in graph.astream(initial, cfg, stream_mode="updates"):
            if not isinstance(update, dict):
                continue
            for node_name, payload in update.items():
                acc.update(payload)
                yield {"event": "graph_step", "node": node_name}
    except Exception as e:
        logger.exception("agentic_rag stream failed: %s", e)
        yield {"event": "error", "detail": str(e)}
        yield {"event": "done"}
        return

    src = acc.get("source") or "general"
    if src not in ("documents", "general"):
        src = "general"
    if src == "documents":
        bd = acc.get("best_dense_dist")
        if bd is not None:
            await record_retrieval_confidence_sample(session, float(bd))
    cites = acc.get("citations") or []
    cite_json = [
        c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in cites
    ]
    yield {
        "event": "meta",
        "citations": cite_json,
        "source": src,
    }
    yield {"event": "token", "text": acc.get("answer") or ""}
    yield {"event": "done"}
