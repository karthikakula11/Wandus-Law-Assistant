import logging
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import UUID

from app.monitoring.openai_usage import record_chat_completion_usage, record_usage_dict
from app.services.drift_detection import record_retrieval_confidence_sample
from app.services.langfuse_tracing import (
    flush_langfuse,
    lf_request_context,
    openai_trace_kwargs,
)
from app.services.openai_factory import get_async_openai_client
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.embeddings import embed_texts
from app.models import Chunk, Document
from app.schemas import CitationOut
from app.services.context_budget import budget_history_messages
from app.services.rag_pipeline import run_generation_phase
from app.services.long_term_memory import normalize_user_key, retrieve_memory_texts
from app.services.rag_prompts import GENERAL_SYSTEM, RAGPromptBuilder, memory_system_suffix
from app.services.small_talk import is_small_talk
from app.services.retrieval_hybrid import retrieve_hybrid_rrf

ReplySource = Literal["documents", "general"]

logger = logging.getLogger(__name__)


def sanitize_history(history: list[dict] | None) -> list[dict]:
    """Keep recent user/assistant turns within message and character budgets."""
    if not history:
        return []
    return budget_history_messages(history)


async def fetch_memory_snippets(
    session: AsyncSession,
    memory_user_id: str | None,
    question: str,
) -> list[str]:
    uk = normalize_user_key(memory_user_id)
    if not uk:
        return []
    try:
        return await retrieve_memory_texts(session, uk, question)
    except Exception as e:
        # e.g. UndefinedTableError when ``alembic upgrade head`` has not been run
        logger.warning("long-term memory retrieval skipped (chat continues): %s", e)
        return []


async def count_chunks(session: AsyncSession) -> int:
    n = await session.scalar(select(func.count()).select_from(Chunk))
    return int(n or 0)


async def retrieve_chunks(
    session: AsyncSession,
    question: str,
    top_k: int,
    *,
    scope_document_ids: list[UUID] | None = None,
) -> list[tuple[Chunk, Document, float]]:
    """Returns (chunk, document, distance) ordered by cosine distance ascending."""
    (qvec,) = await embed_texts([question])
    dist_col = Chunk.embedding.cosine_distance(qvec)

    stmt = (
        select(Chunk, Document, dist_col.label("dist"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(dist_col)
        .limit(top_k)
    )
    if scope_document_ids:
        stmt = stmt.where(Chunk.document_id.in_(scope_document_ids))

    rows = (await session.execute(stmt)).all()
    out: list[tuple[Chunk, Document, float]] = []
    for chunk, doc, dist in rows:
        out.append((chunk, doc, float(dist)))
    return out


def dedupe_retrieved_rows(
    rows: list[tuple[Chunk, Document, float]],
) -> list[tuple[Chunk, Document, float]]:
    """One row per chunk (retrieval can rarely repeat the same chunk id)."""
    seen: set[UUID] = set()
    out: list[tuple[Chunk, Document, float]] = []
    for row in rows:
        cid = row[0].id
        if cid in seen:
            continue
        seen.add(cid)
        out.append(row)
    return out


async def retrieve_for_chat(
    session: AsyncSession,
    question: str,
    top_k: int,
    *,
    scope_document_ids: list[UUID] | None = None,
    use_hybrid: bool | None = None,
) -> tuple[list[tuple[Chunk, Document, float]], float]:
    """Dense-only or hybrid+RRF; returns rows and best dense distance for gating.

    ``use_hybrid`` (Jam API): ``False`` → dense only; ``True``/``None`` → follow ``HYBRID_RAG_ENABLED``.
    """
    settings = get_settings()
    want_hybrid = settings.hybrid_rag_enabled
    if use_hybrid is False:
        want_hybrid = False
    if want_hybrid:
        rows, bd = await retrieve_hybrid_rrf(
            session, question, top_k, scope_document_ids=scope_document_ids
        )
        logger.info(
            "phase=retrieve hybrid=true chunk_count=%s best_dense=%s doc_scope=%s",
            len(rows),
            bd,
            scope_document_ids,
        )
        rows = dedupe_retrieved_rows(rows)
        return rows, bd
    rows = await retrieve_chunks(
        session, question, top_k, scope_document_ids=scope_document_ids
    )
    bd = rows[0][2] if rows else 1.0
    logger.info(
        "phase=retrieve hybrid=false chunk_count=%s best_dense=%s doc_scope=%s",
        len(rows),
        bd,
        scope_document_ids,
    )
    rows = dedupe_retrieved_rows(rows)
    return rows, bd


def build_rag_messages(
    question: str,
    contexts: list[tuple[Chunk, Document]],
    history: list[dict],
    *,
    memory_snippets: list[str] | None = None,
) -> tuple[list[dict], list[tuple[Chunk, Document]]]:
    """Build chat messages from retrieved chunks (delegates to RAGPromptBuilder)."""
    return RAGPromptBuilder().build_messages(
        question, contexts, history, memory_snippets=memory_snippets
    )


async def generate_answer(
    messages: list[dict],
) -> str:
    settings = get_settings()
    client = get_async_openai_client()
    resp = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.65,
        **openai_trace_kwargs(name="rag-answer-generation"),
    )
    await record_chat_completion_usage(resp, route="rag-answer-generation")
    return (resp.choices[0].message.content or "").strip()


async def generate_answer_stream(messages: list[dict]):
    """Async iterator of text deltas from the chat model (course-style /stream)."""
    settings = get_settings()
    client = get_async_openai_client()
    stream = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.65,
        stream=True,
        stream_options={"include_usage": True},
        **openai_trace_kwargs(name="rag-answer-stream"),
    )
    default_model = settings.openai_chat_model
    async for chunk in stream:
        u = getattr(chunk, "usage", None)
        if u is not None:
            m = getattr(chunk, "model", None) or default_model
            await record_usage_dict(
                model=m,
                tokens_in=int(getattr(u, "prompt_tokens", None) or 0),
                tokens_out=int(getattr(u, "completion_tokens", None) or 0),
                route="rag-answer-stream",
            )
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


async def _stream_general_tokens(
    question: str,
    history: list[dict],
    *,
    memory_snippets: list[str] | None = None,
) -> AsyncIterator[str]:
    """Stream tokens for law-only general chat (no RAG excerpts)."""
    settings = get_settings()
    client = get_async_openai_client()
    system = GENERAL_SYSTEM + memory_system_suffix(memory_snippets)
    msgs: list[dict] = [{"role": "system", "content": system}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": question})
    stream = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=msgs,
        temperature=0.65,
        stream=True,
        stream_options={"include_usage": True},
        **openai_trace_kwargs(name="chat-general-stream"),
    )
    default_model = settings.openai_chat_model
    async for chunk in stream:
        u = getattr(chunk, "usage", None)
        if u is not None:
            m = getattr(chunk, "model", None) or default_model
            await record_usage_dict(
                model=m,
                tokens_in=int(getattr(u, "prompt_tokens", None) or 0),
                tokens_out=int(getattr(u, "completion_tokens", None) or 0),
                route="chat-general-stream",
            )
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


async def stream_chat_dispatch(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None = None,
    *,
    scope_document_ids: list[UUID] | None = None,
    langfuse_session_id: str | None = None,
    memory_user_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Server-Sent Events shape: optional ``graph_step`` (LangGraph node id), then ``meta``,
    ``token`` deltas, then ``done``.
    """
    with lf_request_context(langfuse_session_id):
        try:
            mem = await fetch_memory_snippets(session, memory_user_id, question)
            async for ev in _stream_chat_dispatch_inner(
                session,
                question,
                top_k,
                history,
                scope_document_ids=scope_document_ids,
                memory_snippets=mem,
            ):
                yield ev
        finally:
            flush_langfuse()


async def _stream_chat_dispatch_inner(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None,
    *,
    scope_document_ids: list[UUID] | None,
    memory_snippets: list[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    hist = sanitize_history(history)
    mem = memory_snippets or []

    if is_small_talk(question):
        yield {
            "event": "meta",
            "citations": [],
            "source": "general",
        }
        async for t in _stream_general_tokens(question, hist, memory_snippets=mem):
            yield {"event": "token", "text": t}
        yield {"event": "done"}
        return

    n = await count_chunks(session)
    if n == 0:
        yield {"event": "meta", "citations": [], "source": "general"}
        async for t in _stream_general_tokens(question, hist, memory_snippets=mem):
            yield {"event": "token", "text": t}
        yield {"event": "done"}
        return

    settings = get_settings()
    if settings.use_langgraph_agent:
        from app.services import rag_graph

        async for ev in rag_graph.stream_agentic_rag_events(
            session,
            question,
            top_k,
            hist,
            scope_document_ids=scope_document_ids,
            memory_snippets=mem,
        ):
            yield ev
        return

    rows, best_dist = await retrieve_for_chat(
        session, question, top_k, scope_document_ids=scope_document_ids
    )
    if not rows or should_use_general_instead_of_rag(best_dist):
        yield {"event": "meta", "citations": [], "source": "general"}
        async for t in _stream_general_tokens(question, hist, memory_snippets=mem):
            yield {"event": "token", "text": t}
        yield {"event": "done"}
        return

    contexts = [(ch, doc) for ch, doc, _ in rows]
    messages, included = RAGPromptBuilder().build_messages(
        question, contexts, hist, memory_snippets=mem
    )
    cites = citations_from_contexts(included)
    await record_retrieval_confidence_sample(session, float(best_dist))
    yield {
        "event": "meta",
        "citations": [c.model_dump(mode="json") for c in cites],
        "source": "documents",
    }
    async for delta in generate_answer_stream(messages):
        yield {"event": "token", "text": delta}
    yield {"event": "done"}


async def chat_general(
    question: str,
    history: list[dict] | None = None,
    *,
    memory_snippets: list[str] | None = None,
) -> str:
    """Direct LLM reply (gpt-4o-mini) — no RAG retrieval."""
    hist = sanitize_history(history)
    settings = get_settings()
    client = get_async_openai_client()
    system = GENERAL_SYSTEM + memory_system_suffix(memory_snippets)
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(hist)
    messages.append({"role": "user", "content": question})
    resp = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.65,
        **openai_trace_kwargs(name="chat-general"),
    )
    await record_chat_completion_usage(resp, route="chat-general")
    return (resp.choices[0].message.content or "").strip()


def citations_from_contexts(
    contexts: list[tuple[Chunk, Document]],
) -> list[CitationOut]:
    out: list[CitationOut] = []
    seen: set[UUID] = set()
    for ch, doc in contexts:
        if ch.id in seen:
            continue
        seen.add(ch.id)
        excerpt = ch.content[:400] + ("…" if len(ch.content) > 400 else "")
        out.append(
            CitationOut(
                chunk_id=ch.id,
                document_id=doc.id,
                document_title=doc.title,
                chunk_index=ch.chunk_index,
                excerpt=excerpt,
            )
        )
    return out


# Backwards compatibility for tests / imports
MIN_DISTANCE_THRESHOLD = 0.85


def should_use_general_instead_of_rag(best_dense_distance: float) -> bool:
    """
    Jam with AI uses top search hits directly; we only skip RAG when the distance gate is enabled.
    """
    s = get_settings()
    if not s.rag_distance_gate_enabled:
        return False
    return best_dense_distance > s.rag_distance_threshold


async def try_small_talk_general(
    question: str,
    history: list[dict] | None,
    *,
    memory_snippets: list[str] | None = None,
) -> tuple[str, list[CitationOut], ReplySource] | None:
    """Greetings / who-are-you → normal bot chat, not legal excerpt mode."""
    if not is_small_talk(question):
        return None
    hist = history if history is not None else []
    answer = await chat_general(question, hist, memory_snippets=memory_snippets)
    return answer, [], "general"


async def chat_rag_only(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None = None,
    *,
    scope_document_ids: list[UUID] | None = None,
    memory_snippets: list[str] | None = None,
) -> tuple[str, list[CitationOut], ReplySource]:
    """RAG-only path: documents + LLM on excerpts, or abstain."""
    hist = sanitize_history(history)
    mem = memory_snippets or []
    st = await try_small_talk_general(question, hist, memory_snippets=mem)
    if st is not None:
        return st

    settings = get_settings()
    if settings.use_langgraph_agent:
        if await count_chunks(session) == 0:
            return (
                "No documents have been ingested yet, or the knowledge base is empty. "
                "Add text from the Knowledge panel, then ask again.",
                [],
                "documents",
            )
        from app.services import rag_graph

        return await rag_graph.run_agentic_rag(
            session,
            question,
            top_k,
            hist,
            scope_document_ids=scope_document_ids,
            memory_snippets=mem,
        )

    rows, best_dist = await retrieve_for_chat(
        session, question, top_k, scope_document_ids=scope_document_ids
    )
    if not rows:
        return (
            "No documents have been ingested yet, or the knowledge base is empty. "
            "Add text from the Knowledge panel, then ask again.",
            [],
            "documents",
        )

    if should_use_general_instead_of_rag(best_dist):
        answer = await chat_general(question, hist, memory_snippets=mem)
        return answer, [], "general"

    contexts = [(ch, doc) for ch, doc, _ in rows]
    answer, included = await run_generation_phase(
        question, contexts, hist, generate_answer, memory_snippets=mem
    )
    cites = citations_from_contexts(included)
    await record_retrieval_confidence_sample(session, float(best_dist))
    return answer, cites, "documents"


async def chat_auto(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None = None,
    *,
    scope_document_ids: list[UUID] | None = None,
    memory_snippets: list[str] | None = None,
) -> tuple[str, list[CitationOut], ReplySource]:
    """Use indexed law text when confident; otherwise gpt-4o-mini general chat."""
    hist = sanitize_history(history)
    mem = memory_snippets or []
    st = await try_small_talk_general(question, hist, memory_snippets=mem)
    if st is not None:
        return st

    n = await count_chunks(session)
    if n == 0:
        answer = await chat_general(question, hist, memory_snippets=mem)
        return answer, [], "general"

    settings = get_settings()
    if settings.use_langgraph_agent:
        from app.services import rag_graph

        return await rag_graph.run_agentic_rag(
            session,
            question,
            top_k,
            hist,
            scope_document_ids=scope_document_ids,
            memory_snippets=mem,
        )

    rows, best_dist = await retrieve_for_chat(
        session, question, top_k, scope_document_ids=scope_document_ids
    )
    if not rows:
        answer = await chat_general(question, hist, memory_snippets=mem)
        return answer, [], "general"

    if should_use_general_instead_of_rag(best_dist):
        answer = await chat_general(question, hist, memory_snippets=mem)
        return answer, [], "general"

    contexts = [(ch, doc) for ch, doc, _ in rows]
    answer, included = await run_generation_phase(
        question, contexts, hist, generate_answer, memory_snippets=mem
    )
    cites = citations_from_contexts(included)
    await record_retrieval_confidence_sample(session, float(best_dist))
    return answer, cites, "documents"


async def chat_dispatch(
    session: AsyncSession,
    question: str,
    top_k: int,
    history: list[dict] | None = None,
    *,
    scope_document_ids: list[UUID] | None = None,
    langfuse_session_id: str | None = None,
    memory_user_id: str | None = None,
) -> tuple[str, list[CitationOut], ReplySource]:
    """Single path: use uploaded law chunks when retrieval is confident; else general LLM."""
    with lf_request_context(langfuse_session_id):
        try:
            hist = sanitize_history(history)
            mem = await fetch_memory_snippets(session, memory_user_id, question)
            return await chat_auto(
                session,
                question,
                top_k,
                hist,
                scope_document_ids=scope_document_ids,
                memory_snippets=mem,
            )
        finally:
            flush_langfuse()


# Backwards-compatible name for tests
async def chat_rag(
    session: AsyncSession,
    question: str,
    top_k: int,
) -> tuple[str, list[CitationOut]]:
    answer, cites, _ = await chat_rag_only(session, question, top_k, None)
    return answer, cites
