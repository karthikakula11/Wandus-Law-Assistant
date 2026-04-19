"""
Jam with AI course–compatible RAG orchestration.

**Reference (cloned repo)**:
``reference/production-agentic-rag-course/src/routers/ask.py`` — ``_prepare_chunks_and_sources``
(embed if hybrid → ``opensearch_client.search_unified`` → ``RAGPromptBuilder`` → ``ollama_client.generate_rag_answer``),
and ``hybrid_search.py`` for ``search_unified`` listing.

**This app**: ``retrieve_for_chat`` (pgvector + optional BM25 + RRF) replaces ``search_unified``;
``JamRAGPromptBuilder.create_rag_prompt`` matches the Ollama prompt shape; ``generate_jam_openai`` replaces
``OllamaClient.generate_rag_answer`` (same temperature/top_p as ``client.py``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.services.openai_factory import get_async_openai_client
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.monitoring.openai_usage import record_chat_completion_usage, record_usage_dict
from app.jam_schemas import AskRequest, AskResponse, HybridSearchRequest, SearchHit, SearchResponse
from app.services.jam_rag_prompts import JamRAGPromptBuilder
from app.services.langfuse_tracing import flush_langfuse, openai_trace_kwargs
from app.services.rag import retrieve_for_chat

logger = logging.getLogger(__name__)

_JAM_TEMPERATURE = 0.7
_JAM_TOP_P = 0.9


def _rows_to_jam_chunks(
    rows: list[tuple[Any, Any, float]],
) -> list[dict[str, Any]]:
    """Map ORM rows to dicts expected by JamRAGPromptBuilder / course."""
    out: list[dict[str, Any]] = []
    for ch, doc, _score in rows:
        out.append(
            {
                "chunk_text": ch.content,
                "arxiv_id": "",  # law corpus — optional external id
                "document_title": doc.title,
                "chunk_id": str(ch.id),
                "document_id": str(doc.id),
                "source_uri": doc.source_uri,
            }
        )
    return out


def _sources_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for c in chunks:
        uri = c.get("source_uri")
        if uri and str(uri).startswith("http"):
            u = str(uri)
            if u not in seen:
                seen.add(u)
                sources.append(u)
        else:
            label = f"document:{c.get('document_id', '')}"
            if label not in seen:
                seen.add(label)
                sources.append(label)
    return sources


async def generate_jam_openai(prompt: str, model: str) -> str:
    """Course Ollama client uses temperature=0.7, top_p=0.9 on /api/generate."""
    settings = get_settings()
    client = get_async_openai_client()
    resp = await client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=_JAM_TEMPERATURE,
        top_p=_JAM_TOP_P,
        **openai_trace_kwargs(name="jam-ask-openai"),
    )
    await record_chat_completion_usage(resp, route="jam-ask-openai")
    return (resp.choices[0].message.content or "").strip()


async def run_jam_ask(
    session: AsyncSession,
    request: AskRequest,
) -> AskResponse:
    """Mirror course _prepare_chunks_and_sources + RAGPromptBuilder + generate."""
    try:
        rows, _bd = await retrieve_for_chat(
            session,
            request.query,
            request.top_k,
            scope_document_ids=None,
            use_hybrid=request.use_hybrid,
        )
        chunks = _rows_to_jam_chunks(rows)

        if not chunks:
            return AskResponse(
                query=request.query,
                answer=(
                    "I couldn't find any relevant information in the indexed materials "
                    "to answer your question."
                ),
                sources=[],
                chunks_used=0,
                search_mode="bm25" if not request.use_hybrid else "hybrid",
            )

        builder = JamRAGPromptBuilder()
        final_prompt = builder.create_rag_prompt(request.query, chunks)
        answer = await generate_jam_openai(final_prompt, request.model)

        search_mode = "hybrid" if request.use_hybrid else "bm25"

        return AskResponse(
            query=request.query,
            answer=answer,
            sources=_sources_from_chunks(chunks),
            chunks_used=len(chunks),
            search_mode=search_mode,
        )
    finally:
        flush_langfuse()


async def iter_jam_stream(
    session: AsyncSession,
    request: AskRequest,
) -> AsyncIterator[str]:
    """
    Yield SSE lines (already formatted as ``data: ...\\n\\n``) like course stream_router.
    """
    try:
        rows, _bd = await retrieve_for_chat(
            session,
            request.query,
            request.top_k,
            scope_document_ids=None,
            use_hybrid=request.use_hybrid,
        )
        chunks = _rows_to_jam_chunks(rows)
        search_mode = "hybrid" if request.use_hybrid else "bm25"

        if not chunks:
            yield f"data: {json.dumps({'answer': 'No relevant information found.', 'sources': [], 'done': True})}\n\n"
            return

        metadata_response = {
            "sources": _sources_from_chunks(chunks),
            "chunks_used": len(chunks),
            "search_mode": search_mode,
        }
        yield f"data: {json.dumps(metadata_response)}\n\n"

        builder = JamRAGPromptBuilder()
        final_prompt = builder.create_rag_prompt(request.query, chunks)

        settings = get_settings()
        client = get_async_openai_client()
        stream = await client.chat.completions.create(
            model=request.model or settings.openai_chat_model,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=_JAM_TEMPERATURE,
            top_p=_JAM_TOP_P,
            stream=True,
            stream_options={"include_usage": True},
            **openai_trace_kwargs(name="jam-ask-stream"),
        )

        full_response = ""
        default_model = request.model or settings.openai_chat_model
        async for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u is not None:
                m = getattr(chunk, "model", None) or default_model
                await record_usage_dict(
                    model=m,
                    tokens_in=int(getattr(u, "prompt_tokens", None) or 0),
                    tokens_out=int(getattr(u, "completion_tokens", None) or 0),
                    route="jam-ask-stream",
                )
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_response += delta
                yield f"data: {json.dumps({'chunk': delta})}\n\n"

        yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
    finally:
        flush_langfuse()


async def run_jam_hybrid_search(
    session: AsyncSession,
    request: HybridSearchRequest,
) -> SearchResponse:
    """Course hybrid_search router shape; backed by retrieve_for_chat."""
    take = request.size
    rows, _ = await retrieve_for_chat(
        session,
        request.query,
        take,
        scope_document_ids=None,
        use_hybrid=request.use_hybrid,
    )

    mode = "hybrid" if request.use_hybrid else "bm25"
    hits: list[SearchHit] = []
    for ch, doc, score in rows:
        if score < request.min_score:
            continue
        pdf_url = None
        if doc.source_uri and str(doc.source_uri).startswith("http"):
            pdf_url = str(doc.source_uri)
        hits.append(
            SearchHit(
                arxiv_id="",
                title=doc.title,
                authors=None,
                abstract=None,
                published_date=None,
                pdf_url=pdf_url,
                score=float(score),
                chunk_text=ch.content,
                chunk_id=str(ch.id),
            )
        )

    return SearchResponse(
        query=request.query,
        total=len(hits),
        hits=hits,
        size=request.size,
        from_=request.from_,
        search_mode=mode,
    )
