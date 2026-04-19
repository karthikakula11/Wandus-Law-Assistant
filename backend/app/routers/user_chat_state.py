"""Persist full chat sidebar (Recents) per ``memory_user_id`` — survives restarts / URL changes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas import ChatThreadsStateIn
from app.services.long_term_memory import normalize_user_key
from app.services.user_chat_state import get_chat_state_payload, upsert_chat_state_payload

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/chat-state", response_model=ChatThreadsStateIn)
async def get_user_chat_state(
    memory_user_id: str = Query(..., min_length=8, max_length=64),
    session: AsyncSession = Depends(get_session),
) -> ChatThreadsStateIn:
    uk = normalize_user_key(memory_user_id)
    if not uk:
        raise HTTPException(status_code=400, detail="invalid memory_user_id")
    raw = await get_chat_state_payload(session, uk)
    await session.commit()
    if raw is None:
        raise HTTPException(status_code=404, detail="not_found")
    try:
        return ChatThreadsStateIn.model_validate(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail="stored chat state is invalid") from e


@router.put("/chat-state", status_code=204)
async def put_user_chat_state(
    body: ChatThreadsStateIn,
    memory_user_id: str = Query(..., min_length=8, max_length=64),
    session: AsyncSession = Depends(get_session),
) -> None:
    uk = normalize_user_key(memory_user_id)
    if not uk:
        raise HTTPException(status_code=400, detail="invalid memory_user_id")
    await upsert_chat_state_payload(session, uk, body.model_dump(mode="json"))
    await session.commit()
