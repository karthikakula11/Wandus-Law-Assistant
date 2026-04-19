"""Load / save full client chat sidebar state (threads + messages) per user key."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserChatState


async def get_chat_state_payload(session: AsyncSession, user_key: str) -> dict | None:
    row = await session.scalar(
        select(UserChatState).where(UserChatState.memory_user_key == user_key).limit(1)
    )
    if row is None:
        return None
    return dict(row.payload) if isinstance(row.payload, dict) else None


async def upsert_chat_state_payload(session: AsyncSession, user_key: str, payload: dict) -> None:
    row = await session.scalar(
        select(UserChatState).where(UserChatState.memory_user_key == user_key).limit(1)
    )
    now = datetime.now(timezone.utc)
    if row is None:
        session.add(
            UserChatState(memory_user_key=user_key, payload=payload, updated_at=now)
        )
    else:
        row.payload = payload
        row.updated_at = now
    await session.flush()
