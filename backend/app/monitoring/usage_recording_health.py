"""
Orchestrator-style health report for token + cost recording.

Aggregates: Langfuse pricing cache, last refresh, ``llm_usage_log`` null-cost rows,
recent volume, and actionable recommendations (no separate LLM — pure rules + SQL).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import LLMUsageLog
from app.monitoring import llm_pricing


async def build_usage_recording_health(session: AsyncSession) -> dict[str, Any]:
    """Return structured status for monitoring dashboards and alerts."""
    now = datetime.now(timezone.utc)
    since_7d = now - timedelta(days=7)
    since_24h = now - timedelta(hours=24)

    pricing = llm_pricing.get_pricing_status()
    models_cached = int(pricing.get("models_cached") or 0)
    lf_ok = bool(pricing.get("langfuse_configured"))

    total_rows = int(
        (await session.execute(select(func.count()).select_from(LLMUsageLog))).scalar() or 0
    )
    rows_7d = int(
        (
            await session.execute(
                select(func.count()).select_from(LLMUsageLog).where(LLMUsageLog.created_at >= since_7d)
            )
        ).scalar()
        or 0
    )
    rows_24h = int(
        (
            await session.execute(
                select(func.count()).select_from(LLMUsageLog).where(LLMUsageLog.created_at >= since_24h)
            )
        ).scalar()
        or 0
    )

    null_cost_7d = int(
        (
            await session.execute(
                select(func.count())
                .select_from(LLMUsageLog)
                .where(
                    LLMUsageLog.cost_usd.is_(None),
                    LLMUsageLog.created_at >= since_7d,
                )
            )
        ).scalar()
        or 0
    )

    null_by_model_rows = (
        (
            await session.execute(
                select(LLMUsageLog.model, func.count(LLMUsageLog.id))
                .where(LLMUsageLog.cost_usd.is_(None), LLMUsageLog.created_at >= since_7d)
                .group_by(LLMUsageLog.model)
                .order_by(func.count(LLMUsageLog.id).desc())
                .limit(20)
            )
        )
        .all()
    )
    null_cost_by_model = [
        {"model": str(m), "rows": int(c)} for m, c in null_by_model_rows if m
    ]

    route_rows = (
        (
            await session.execute(
                select(LLMUsageLog.route, func.count(LLMUsageLog.id))
                .where(LLMUsageLog.created_at >= since_7d)
                .group_by(LLMUsageLog.route)
                .order_by(func.count(LLMUsageLog.id).desc())
                .limit(30)
            )
        )
        .all()
    )
    rows_by_route_7d = [
        {"route": str(r or "unknown"), "rows": int(c)} for r, c in route_rows
    ]

    recommendations: list[str] = []
    status: str = "ok"

    if not lf_ok:
        recommendations.append(
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY so pricing can load from "
            "GET /api/public/models and OpenAI calls use the Langfuse-wrapped client."
        )
        if rows_7d > 0:
            status = "degraded"
    elif models_cached == 0:
        recommendations.append(
            "Langfuse is configured but the pricing cache is empty — check network, "
            "credentials, and LANGFUSE_BASE_URL; restart the API or wait for the next refresh."
        )
        if rows_7d > 0:
            status = "critical"

    if pricing.get("last_refresh_error"):
        recommendations.append(
            f"Last pricing refresh failed: {pricing['last_refresh_error'][:500]}"
        )
        if status == "ok":
            status = "degraded"

    if null_cost_7d > 0:
        recommendations.append(
            f"In the last 7 days, {null_cost_7d} usage row(s) have tokens but cost_usd is NULL — "
            "model may be missing from Langfuse catalog; try POST /monitoring/usage/sync-costs "
            "after fixing pricing, or add the model in Langfuse."
        )
        status = "degraded" if status == "ok" else status

    if rows_7d == 0 and total_rows == 0:
        recommendations.append("No usage rows yet — send chat or run ingest to verify recording.")

    if status == "ok" and not recommendations:
        recommendations.append("Token and cost recording pipeline looks healthy for current checks.")

    return {
        "service": "usage_recording_orchestrator",
        "checked_at": now.isoformat(),
        "status": status,
        "pricing": pricing,
        "llm_usage_log": {
            "total_rows": total_rows,
            "rows_last_24h": rows_24h,
            "rows_last_7d": rows_7d,
            "null_cost_rows_last_7d": null_cost_7d,
            "null_cost_by_model_last_7d": null_cost_by_model,
            "rows_by_route_last_7d": rows_by_route_7d,
        },
        "checks": {
            "langfuse_keys_configured": lf_ok,
            "pricing_catalog_has_models": models_cached > 0,
            "no_recent_null_cost_rows": null_cost_7d == 0,
        },
        "recommendations": recommendations,
    }
