"""CRUD for long-term memory items (per ``memory_user_id``)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.schemas import MemoryItemCreate, MemoryItemOut, MemoryListResponse
from app.services.long_term_memory import (
    add_memory_item,
    delete_memory_item,
    list_memory_items,
    normalize_user_key,
)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("/items", response_model=MemoryListResponse)
async def get_memory_items(
    memory_user_id: str = Query(..., min_length=8, max_length=64),
    session: AsyncSession = Depends(get_session),
) -> MemoryListResponse:
    uk = normalize_user_key(memory_user_id)
    if not uk:
        raise HTTPException(status_code=400, detail="invalid memory_user_id")
    rows = await list_memory_items(session, uk)
    await session.commit()
    return MemoryListResponse(
        items=[
            MemoryItemOut(id=r.id, content=r.content, created_at=r.created_at) for r in rows
        ]
    )


@router.post("/items", response_model=MemoryItemOut)
async def post_memory_item(
    body: MemoryItemCreate,
    session: AsyncSession = Depends(get_session),
) -> MemoryItemOut:
    uk = normalize_user_key(body.memory_user_id)
    if not uk:
        raise HTTPException(status_code=400, detail="invalid memory_user_id")
    try:
        row = await add_memory_item(session, uk, body.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    await session.commit()
    await session.refresh(row)
    return MemoryItemOut(id=row.id, content=row.content, created_at=row.created_at)


@router.delete("/items/{item_id}", status_code=204)
async def remove_memory_item(
    item_id: UUID,
    memory_user_id: str = Query(..., min_length=8, max_length=64),
    session: AsyncSession = Depends(get_session),
) -> None:
    uk = normalize_user_key(memory_user_id)
    if not uk:
        raise HTTPException(status_code=400, detail="invalid memory_user_id")
    ok = await delete_memory_item(session, uk, item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not_found")
    await session.commit()
