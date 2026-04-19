"""
POST ``/api/public/ingestion`` with trace-create + generation-create.

Omits cost in the payload so Langfuse applies model pricing from usageDetails (tokens).

**Default:** disabled via ``LANGFUSE_MANUAL_INGESTION=false`` because the Langfuse OpenAI
wrapper already records generations; enabling this duplicates Langfuse observations unless
you use plain OpenAI only.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import get_settings
from app.monitoring.langfuse_config import get_langfuse_config, langfuse_enabled

logger = logging.getLogger(__name__)


def _iso_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


async def write_langfuse_usage_sync(
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    session_id: str | None,
    route: str,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """
    Send one trace + one generation to Langfuse ingestion. Returns trace id, or None if skipped/failed.
    """
    s = get_settings()
    if not s.langfuse_manual_ingestion:
        return None
    if not langfuse_enabled():
        return None
    cfg = get_langfuse_config()
    if not cfg:
        return None

    trace_id = str(uuid.uuid4())
    gen_id = str(uuid.uuid4())
    ts = _iso_ts()
    total = max(0, tokens_in + tokens_out)
    body_trace: dict[str, Any] = {
        "id": trace_id,
        "name": route or "pintu-llm",
        "metadata": metadata or {},
    }
    if session_id:
        body_trace["sessionId"] = session_id

    body_gen: dict[str, Any] = {
        "id": gen_id,
        "traceId": trace_id,
        "name": route or "chat-completion",
        "startTime": ts,
        "endTime": ts,
        "model": model,
        "usageDetails": {
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "total_tokens": total,
        },
    }

    batch = [
        {
            "type": "trace-create",
            "id": str(uuid.uuid4()),
            "timestamp": ts,
            "body": body_trace,
        },
        {
            "type": "generation-create",
            "id": str(uuid.uuid4()),
            "timestamp": ts,
            "body": body_gen,
        },
    ]

    url = f"{cfg.base_url}/api/public/ingestion"
    headers = {
        "Authorization": f"Basic {cfg.basic_auth_header()}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, headers=headers, json={"batch": batch})
            if r.status_code >= 400:
                logger.warning("langfuse ingestion HTTP %s: %s", r.status_code, r.text[:500])
                return None
    except Exception as e:
        logger.warning("langfuse ingestion failed: %s", e)
        return None
    return trace_id
