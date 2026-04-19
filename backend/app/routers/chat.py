import json
import logging
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session
from app.schemas import ChatBody, ChatResponse
from app.services.conversation_memory import persist_auto_memory_from_turn
from app.services.document_router import prepare_retrieval_scope
from app.services.rag import chat_dispatch, stream_chat_dispatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatBody,
    session: AsyncSession = Depends(get_session),
):
    hist = [h.model_dump() for h in body.history] if body.history else None
    scope_ids = await prepare_retrieval_scope(
        session,
        body.question,
        hist,
        body.document_id,
    )
    lf_sid = str(uuid.uuid4())
    answer, citations, source = await chat_dispatch(
        session,
        question=body.question,
        top_k=body.top_k,
        history=hist,
        scope_document_ids=scope_ids,
        langfuse_session_id=lf_sid,
        memory_user_id=body.memory_user_id,
    )
    if body.auto_memory and get_settings().auto_memory_from_conversation and answer.strip():
        try:
            n = await persist_auto_memory_from_turn(
                session, body.memory_user_id, body.question, answer
            )
            if n:
                await session.commit()
        except Exception:
            logger.exception("auto_memory after non-stream chat failed")

    return ChatResponse(answer=answer, citations=citations, source=source)


@router.post("/stream")
async def chat_stream(
    body: ChatBody,
    session: AsyncSession = Depends(get_session),
):
    """
    Server-Sent Events (SSE): `data: {"event":"meta",...}` then `{"event":"token","text":...}` ... `{"event":"done"}`.
    Mirrors course-style `/ask` streaming over OpenSearch + Ollama; here: same scope/routing as `POST /chat`.
    """

    async def event_stream():
        settings = get_settings()
        do_auto_memory = (
            body.auto_memory
            and settings.auto_memory_from_conversation
        )
        saw_error = False
        token_parts: list[str] = []

        hist = [h.model_dump() for h in body.history] if body.history else None
        scope_ids = await prepare_retrieval_scope(
            session,
            body.question,
            hist,
            body.document_id,
        )
        lf_sid = str(uuid.uuid4())
        async for ev in stream_chat_dispatch(
            session,
            question=body.question,
            top_k=body.top_k,
            history=hist,
            scope_document_ids=scope_ids,
            langfuse_session_id=lf_sid,
            memory_user_id=body.memory_user_id,
        ):
            if ev.get("event") == "error":
                saw_error = True
            if ev.get("event") == "token" and ev.get("text"):
                token_parts.append(ev["text"])
            yield f"data: {json.dumps(ev)}\n\n"

        answer = "".join(token_parts)
        if do_auto_memory and not saw_error and answer.strip():
            try:
                n = await persist_auto_memory_from_turn(
                    session, body.memory_user_id, body.question, answer
                )
                if n:
                    await session.commit()
            except Exception:
                logger.exception("auto_memory after stream failed")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
