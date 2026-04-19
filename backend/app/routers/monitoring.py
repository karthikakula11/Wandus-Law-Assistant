"""Monitoring: local usage summary, Langfuse dashboard (TalentVibe-style metrics), cost sync."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import LLMUsageLog
from app.monitoring import llm_pricing
from app.monitoring.langfuse_cost_sync import sync_company_costs
from app.monitoring.langfuse_dashboard import build_langfuse_costs_response
from app.monitoring.usage_recording_health import build_usage_recording_health

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


@router.get("/usage/recording-health")
async def usage_recording_health(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    """
    Orchestrator report: pricing cache, recent ``llm_usage_log`` volume, null-cost rows,
    per-model/route hints, and recommendations. Use for alerts or the Usage UI.
    """
    return await build_usage_recording_health(session)


@router.get("/usage/summary")
async def usage_summary(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    """Aggregate tokens and cost from ``llm_usage_log`` (local DB)."""
    row = (
        await session.execute(
            select(
                func.coalesce(func.sum(LLMUsageLog.tokens_in), 0),
                func.coalesce(func.sum(LLMUsageLog.tokens_out), 0),
                func.coalesce(func.sum(LLMUsageLog.cost_usd), 0.0),
                func.count(LLMUsageLog.id),
            )
        )
    ).one()
    tin, tout, cost, n = row
    pricing = llm_pricing.get_pricing_status()
    return {
        "rows": int(n),
        "input_tokens": int(tin),
        "output_tokens": int(tout),
        "total_tokens": int(tin) + int(tout),
        "cost_usd_sum": float(cost or 0.0),
        "pricing": pricing,
        "note": (
            "Tokens from OpenAI usage fields; estimated USD = tokens × Langfuse "
            "GET /api/public/models inputPrice/outputPrice per token. "
            "Includes chat, streaming, and embed-texts rows."
        ),
    }


@router.post("/usage/sync-costs")
async def post_sync_costs(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    """Backfill ``cost_usd``: recompute from model catalog, then Langfuse trace totalCost."""
    try:
        out = await sync_company_costs(session, limit=limit)
    except Exception as e:
        logger.warning("sync-costs failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return out


@router.get("/langfuse/costs")
async def get_langfuse_costs(
    start_date: Optional[str] = Query(None, description="ISO start (optional)"),
    end_date: Optional[str] = Query(None, description="ISO end (optional)"),
    days: int = Query(30, ge=1, le=365),
    session_id: Optional[str] = Query(
        None,
        description="Optional Langfuse sessionId filter; omit for project-wide totals",
    ),
) -> dict[str, Any]:
    """
    TalentVibe / fix_langfuse-style dashboard: traces + ``/api/public/metrics`` (observations)
    for input/output/total tokens and cost, with short TTL cache and 429 retry cap.
    """
    return await build_langfuse_costs_response(
        start_date=start_date,
        end_date=end_date,
        days=days,
        session_id=session_id,
    )
