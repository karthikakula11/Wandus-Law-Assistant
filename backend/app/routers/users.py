"""Read-only user endpoints (seeded primary user for memory key)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import User
from app.schemas import UserOut

router = APIRouter(prefix="/users", tags=["users"])

# Matches alembic seed `004_users_seed.py`
PRIMARY_USERNAME = "owner"


@router.get("/primary", response_model=UserOut)
async def get_primary_user(session: AsyncSession = Depends(get_session)) -> UserOut:
    """Return the seeded **owner** user (``username=owner``) for default memory/chat key."""
    row = await session.scalar(select(User).where(User.username == PRIMARY_USERNAME).limit(1))
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No primary user — run database migrations (alembic upgrade head).",
        )
    return UserOut(
        id=row.id,
        username=row.username,
        display_name=row.display_name,
        memory_user_key=row.memory_user_key,
        created_at=row.created_at,
    )
