"""Minimal trace-only events (e.g. auth/session markers) without LLM usage."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from app.monitoring.langfuse_config import get_langfuse_config, langfuse_enabled

logger = logging.getLogger(__name__)


def _iso_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


async def ingest_trace_event(
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> str | None:
    """Send a single trace-create. Returns trace id or None."""
    if not langfuse_enabled():
        return None
    cfg = get_langfuse_config()
    if not cfg:
        return None

    trace_id = str(uuid.uuid4())
    ts = _iso_ts()
    body: dict[str, Any] = {"id": trace_id, "name": name, "metadata": metadata or {}}
    if session_id:
        body["sessionId"] = session_id
    batch = [{"type": "trace-create", "id": str(uuid.uuid4()), "timestamp": ts, "body": body}]
    url = f"{cfg.base_url}/api/public/ingestion"
    headers = {
        "Authorization": f"Basic {cfg.basic_auth_header()}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json={"batch": batch})
            if r.status_code >= 400:
                logger.warning("ingest_trace_event HTTP %s", r.status_code)
                return None
    except Exception as e:
        logger.warning("ingest_trace_event: %s", e)
        return None
    return trace_id
