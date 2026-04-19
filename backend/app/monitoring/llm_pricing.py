"""
In-memory pricing from Langfuse ``GET /api/public/models`` (USD per token for input/output).

Token counts come from OpenAI responses; this module supplies per-token USD rates from Langfuse’s
public model catalog. ``compute_cost`` = tokens_in * inputPrice + tokens_out * outputPrice
(embedding rows use output tokens 0).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import get_settings
from app.monitoring.langfuse_config import get_langfuse_config, langfuse_enabled

logger = logging.getLogger(__name__)

# model_name (lowercase) -> (usd_per_input_token, usd_per_output_token)
_pricing: dict[str, tuple[float, float]] = {}
_lock = asyncio.Lock()
_last_pricing_refresh_at: datetime | None = None
_last_pricing_refresh_error: str | None = None


def _normalize_model(name: str) -> str:
    return (name or "").strip().lower()


def _strip_openai_model_version_suffix(name: str) -> str:
    """Map ``gpt-4o-mini-2024-07-18``-style strings toward catalog keys like ``gpt-4o-mini``."""
    s = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    s = re.sub(r"-\d{4}$", "", s)
    return s


def _candidate_model_keys(model: str) -> list[str]:
    """Ordered keys to try against the Langfuse pricing map."""
    key = _normalize_model(model)
    if not key:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for k in (key, _strip_openai_model_version_suffix(key)):
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


async def refresh_from_langfuse() -> None:
    """Paginate ``/api/public/models`` and fill the in-memory pricing table."""
    global _last_pricing_refresh_at, _last_pricing_refresh_error

    if not langfuse_enabled():
        return
    cfg = get_langfuse_config()
    if not cfg:
        return
    url = f"{cfg.base_url}/api/public/models"
    headers = {"Authorization": f"Basic {cfg.basic_auth_header()}"}
    new_map: dict[str, tuple[float, float]] = {}
    page = 1
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                r = await client.get(url, headers=headers, params={"page": page, "limit": 100})
                r.raise_for_status()
                body = r.json()
                rows = body.get("data") or []
                for m in rows:
                    name = str(m.get("modelName") or "").strip()
                    if not name:
                        continue
                    inp = m.get("inputPrice")
                    out = m.get("outputPrice")
                    if inp is None and out is None:
                        continue
                    ti = float(inp or 0.0)
                    to = float(out or 0.0)
                    new_map[_normalize_model(name)] = (ti, to)
                meta = body.get("meta") or {}
                total_pages = int(meta.get("totalPages") or 1)
                if page >= total_pages:
                    break
                page += 1

        async with _lock:
            _pricing.clear()
            _pricing.update(new_map)
        _last_pricing_refresh_at = datetime.now(timezone.utc)
        _last_pricing_refresh_error = None
        logger.info("llm_pricing refreshed: %s models", len(_pricing))
    except Exception as e:
        _last_pricing_refresh_error = str(e)
        raise


async def ensure_pricing_cached() -> None:
    """Load Langfuse model prices if enabled and cache is empty (e.g. startup refresh failed)."""
    if not langfuse_enabled():
        return
    if pricing_cache_size() > 0:
        return
    try:
        await refresh_from_langfuse()
    except Exception as e:
        logger.warning("ensure_pricing_cached: could not load Langfuse model prices: %s", e)


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """
    Return estimated USD cost using cached per-token rates, or None if model unknown.

    Langfuse ``inputPrice`` / ``outputPrice`` are USD **per token** for matched models.
    """
    rates: tuple[float, float] | None = None
    for key in _candidate_model_keys(model):
        rates = _pricing.get(key)
        if rates:
            break
    if not rates:
        for key in _candidate_model_keys(model):
            for k, v in _pricing.items():
                if key in k or k in key:
                    rates = v
                    break
            if rates:
                break
    if not rates:
        return None
    inp_usd, out_usd = rates
    return float(tokens_in) * inp_usd + float(tokens_out) * out_usd


def pricing_cache_size() -> int:
    return len(_pricing)


def get_pricing_status() -> dict[str, Any]:
    """Metadata for monitoring/UI: where rates come from and whether cache is populated."""
    return {
        "source": "langfuse_get_api_public_models",
        "models_cached": pricing_cache_size(),
        "langfuse_configured": langfuse_enabled(),
        "cost_formula": "tokens_in * inputPrice + tokens_out * outputPrice (USD per token)",
        "last_refresh_at": _last_pricing_refresh_at.isoformat() if _last_pricing_refresh_at else None,
        "last_refresh_error": _last_pricing_refresh_error,
    }


async def pricing_refresh_loop() -> None:
    """Background loop: sleep first (startup already refreshed), then periodic ``refresh_from_langfuse``."""
    while True:
        hours = float(get_settings().langfuse_pricing_refresh_hours)
        await asyncio.sleep(max(60.0, hours * 3600.0))
        try:
            await refresh_from_langfuse()
        except Exception as e:
            logger.warning("llm_pricing refresh failed: %s", e)
