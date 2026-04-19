"""Persist LLM token usage + estimated cost (local pricing cache); optional Langfuse HTTP mirror."""

from __future__ import annotations

import logging

from app.config import get_settings
from app.database import get_session_factory
from app.models import LLMUsageLog
from app.monitoring import llm_pricing
from app.monitoring.llm_usage_writer import write_langfuse_usage_sync

logger = logging.getLogger(__name__)


async def record_llm_usage(
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    session_id: str | None,
    route: str,
) -> None:
    """Insert ``LLMUsageLog``; optionally mirror to Langfuse ingestion when enabled."""
    if tokens_in < 0 or tokens_out < 0:
        return
    if tokens_in == 0 and tokens_out == 0:
        return
    s = get_settings()
    manual_tid = None
    if s.langfuse_manual_ingestion:
        manual_tid = await write_langfuse_usage_sync(
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            session_id=session_id,
            route=route,
        )
    await llm_pricing.ensure_pricing_cached()
    cost = llm_pricing.compute_cost(model, tokens_in, tokens_out)
    factory = get_session_factory()
    try:
        async with factory() as db:
            db.add(
                LLMUsageLog(
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost,
                    trace_id=manual_tid,
                    session_id=session_id,
                    route=route,
                )
            )
            await db.commit()
    except Exception as e:
        logger.warning("llm usage log failed: %s", e)
