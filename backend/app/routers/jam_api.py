"""
Jam with AI course–compatible HTTP API.

**Reference (cloned repo)**:
``reference/production-agentic-rag-course/src/routers/ask.py`` (``ask_router``, ``stream_router``),
``src/routers/hybrid_search.py`` — mounted under ``/api/v1`` in ``src/main.py`` lines 114–117.

This router: ``POST /api/v1/ask``, ``POST /api/v1/stream``, ``POST /api/v1/hybrid-search``.
Generation uses OpenAI (GPT) instead of Ollama; ``model`` is the chat model name.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.jam_schemas import AskRequest, AskResponse, HybridSearchRequest, SearchResponse
from app.services.jam_ask_service import iter_jam_stream, run_jam_ask, run_jam_hybrid_search

router = APIRouter(tags=["jam-course"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    session: AsyncSession = Depends(get_session),
) -> AskResponse:
    return await run_jam_ask(session, request)


@router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    async def generate_stream():
        async for line in iter_jam_stream(session, request):
            yield line

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/hybrid-search", response_model=SearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    session: AsyncSession = Depends(get_session),
) -> SearchResponse:
    return await run_jam_hybrid_search(session, request)
