"""
Backfill ``cost_usd`` on ``LLMUsageLog``: (1) re-run ``compute_cost`` from model + tokens;
(2) for rows still NULL, fetch ``GET /api/public/traces/{trace_id}`` and read ``totalCost``.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.monitoring.langfuse_config import get_langfuse_config, langfuse_enabled
from app.monitoring import llm_pricing
from app.models import LLMUsageLog

logger = logging.getLogger(__name__)


async def sync_company_costs(session: AsyncSession, *, limit: int = 500) -> dict[str, Any]:
    """Pass 1: local pricing; pass 2: Langfuse trace ``totalCost`` fallback."""
    n1 = await _pass_recompute_local(session, limit=limit)
    n2 = 0
    if langfuse_enabled():
        n2 = await _pass_fetch_trace_cost(session, limit=limit)
    return {"recomputed_from_pricing": n1, "filled_from_langfuse_trace": n2}


async def _pass_recompute_local(session: AsyncSession, *, limit: int) -> int:
    stmt = (
        select(LLMUsageLog)
        .where(LLMUsageLog.cost_usd.is_(None))
        .order_by(LLMUsageLog.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    n = 0
    for row in rows:
        c = llm_pricing.compute_cost(row.model, row.tokens_in, row.tokens_out)
        if c is None:
            continue
        await session.execute(
            update(LLMUsageLog).where(LLMUsageLog.id == row.id).values(cost_usd=c)
        )
        n += 1
    await session.commit()
    return n


async def _pass_fetch_trace_cost(session: AsyncSession, *, limit: int) -> int:
    cfg = get_langfuse_config()
    if not cfg:
        return 0
    stmt = (
        select(LLMUsageLog)
        .where(LLMUsageLog.cost_usd.is_(None))
        .where(LLMUsageLog.trace_id.isnot(None))
        .order_by(LLMUsageLog.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    headers = {"Authorization": f"Basic {cfg.basic_auth_header()}"}
    n = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for row in rows:
            tid = row.trace_id
            if not tid:
                continue
            url = f"{cfg.base_url}/api/public/traces/{tid}"
            try:
                r = await client.get(url, headers=headers)
                if r.status_code != 200:
                    continue
                body = r.json()
                tc = body.get("totalCost")
                if tc is None:
                    continue
                await session.execute(
                    update(LLMUsageLog)
                    .where(LLMUsageLog.id == row.id)
                    .values(cost_usd=float(tc))
                )
                n += 1
            except Exception as e:
                logger.debug("trace cost fetch skip %s: %s", tid, e)
    await session.commit()
    return n
