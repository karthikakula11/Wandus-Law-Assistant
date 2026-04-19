"""
Langfuse cost/token dashboard (TalentVibe-style).

- Traces: ``GET /api/public/traces`` (paginated) for ``totalCost`` sum + trace_count.
- Metrics: ``GET /api/public/metrics`` with ``view=observations`` for input/output/total tokens
  and cost (same approach as fix_langfuse / TalentVibe monitoring API).

Optional ``session_id`` filters to one Langfuse session; omit for **project-wide** totals
(Wandus has no company_id — each chat uses a random session UUID unless you pass a scope).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from app.monitoring.langfuse_config import get_langfuse_config, langfuse_enabled

logger = logging.getLogger(__name__)

LANGFUSE_COSTS_CACHE_TTL_SECONDS = 8
LANGFUSE_METRICS_MAX_RETRY_WAIT_SECONDS = 2.0
MAX_TRACE_PAGES = 5
TRACE_PAGE_LIMIT = 100

_langfuse_costs_cache: dict[str, dict[str, Any]] = {}
_langfuse_last_success: dict[str, dict[str, Any]] = {}


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return int(float(value or 0))


def _metric_raw(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            value = row.get(key)
            if isinstance(value, dict):
                if "value" in value:
                    return value.get("value")
                if "sum" in value:
                    return value.get("sum")
            return value
    return None


def _metric_int(row: dict[str, Any], *keys: str) -> int:
    raw = _metric_raw(row, *keys)
    if raw is None:
        return 0
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def _metric_float(row: dict[str, Any], *keys: str) -> float:
    raw = _metric_raw(row, *keys)
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        iso_str = dt_str.replace("Z", "+00:00") if dt_str.endswith("Z") else dt_str
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _round_dt_for_cache(dt: datetime, bucket_seconds: int = 10) -> datetime:
    dt_utc = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    epoch = int(dt_utc.timestamp())
    rounded_epoch = (epoch // bucket_seconds) * bucket_seconds
    return datetime.fromtimestamp(rounded_epoch, tz=timezone.utc)


def _langfuse_cache_key(scope: str, start_dt: datetime, end_dt: datetime) -> str:
    start_iso = (start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)).isoformat()
    end_iso = _round_dt_for_cache(end_dt).isoformat()
    return f"{scope}|{start_iso}|{end_iso}"


def _get_langfuse_cached_payload(cache_key: str) -> Optional[dict[str, Any]]:
    entry = _langfuse_costs_cache.get(cache_key)
    if not entry:
        return None
    if entry.get("expires_at") <= datetime.now(timezone.utc):
        _langfuse_costs_cache.pop(cache_key, None)
        return None
    return entry.get("payload")


def _set_langfuse_cached_payload(cache_key: str, payload: dict[str, Any]) -> None:
    _langfuse_costs_cache[cache_key] = {
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=LANGFUSE_COSTS_CACHE_TTL_SECONDS),
        "payload": payload,
    }


def _utc_iso_range(
    start_date: Optional[str], end_date: Optional[str], days: int
) -> tuple[datetime, datetime, str, str]:
    """Return (start_naive_utc, end_naive_utc, from_ts, to_ts) for Langfuse APIs."""
    end_parsed = _parse_iso(end_date)
    if end_parsed is None:
        end_aware = datetime.now(timezone.utc)
    else:
        end_aware = (
            end_parsed.replace(tzinfo=timezone.utc)
            if end_parsed.tzinfo is None
            else end_parsed.astimezone(timezone.utc)
        )
    start_parsed = _parse_iso(start_date)
    if start_parsed is None:
        start_aware = end_aware - timedelta(days=days)
    else:
        start_aware = (
            start_parsed.replace(tzinfo=timezone.utc)
            if start_parsed.tzinfo is None
            else start_parsed.astimezone(timezone.utc)
        )
    from_ts = start_aware.isoformat().replace("+00:00", "Z")
    to_ts = end_aware.isoformat().replace("+00:00", "Z")
    start_naive = start_aware.replace(tzinfo=None)
    end_naive = end_aware.replace(tzinfo=None)
    return start_naive, end_naive, from_ts, to_ts


async def _langfuse_get_with_retry(
    client: httpx.AsyncClient,
    *,
    url: str,
    auth: tuple[str, str],
    params: dict[str, Any],
    max_attempts: int = 2,
) -> httpx.Response:
    response: httpx.Response | None = None
    for attempt in range(max_attempts):
        response = await client.get(url, auth=auth, params=params)
        if response.status_code != 429:
            return response
        retry_after = _safe_int(response.headers.get("Retry-After"))
        if retry_after > int(LANGFUSE_METRICS_MAX_RETRY_WAIT_SECONDS):
            logger.warning(
                "Langfuse metrics rate-limited (429) with large Retry-After=%ss; skipping retries.",
                retry_after,
            )
            return response
        backoff = float(retry_after) if retry_after > 0 else min(
            LANGFUSE_METRICS_MAX_RETRY_WAIT_SECONDS,
            0.2 * (2**attempt),
        )
        logger.warning(
            "Langfuse metrics rate-limited (429). attempt=%s/%s wait=%ss",
            attempt + 1,
            max_attempts,
            backoff,
        )
        if attempt < max_attempts - 1:
            await asyncio.sleep(backoff)
    assert response is not None
    return response


def _session_filters(session_id: str | None) -> list[dict[str, Any]]:
    if not session_id:
        return []
    return [
        {
            "column": "sessionId",
            "operator": "=",
            "value": str(session_id),
            "type": "string",
        }
    ]


async def build_langfuse_costs_response(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: int = 30,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Return ``{ success, data }`` in TalentVibe / fix_langfuse shape."""
    if not langfuse_enabled():
        return {
            "success": True,
            "data": {
                "enabled": False,
                "message": "Langfuse is not configured (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY).",
                "totals": {},
                "trends": [],
                "models": [],
            },
        }

    cfg = get_langfuse_config()
    if not cfg:
        return {
            "success": True,
            "data": {
                "enabled": False,
                "message": "Langfuse config missing.",
                "totals": {},
                "trends": [],
                "models": [],
            },
        }

    start_dt, end_dt, from_ts, to_ts = _utc_iso_range(start_date, end_date, days)
    scope_key = session_id or "project"
    cache_key = _langfuse_cache_key(scope_key, start_dt, end_dt)
    cached = _get_langfuse_cached_payload(cache_key)
    if cached:
        cached_copy = json.loads(json.dumps(cached))
        cached_copy.setdefault("meta", {})["cached"] = True
        cached_copy["meta"]["last_served_at"] = datetime.now(timezone.utc).isoformat()
        return {"success": True, "data": cached_copy}

    auth = (cfg.public_key, cfg.secret_key)
    base = cfg.base_url

    traces: list[dict[str, Any]] = []
    total_items = 0
    pages_fetched = 0
    page = 1

    totals_metrics_status: int | None = None
    trends_metrics_status: int | None = None
    models_metrics_status: int | None = None
    totals_metrics_sample: dict[str, Any] | None = None
    models_metrics_sample: dict[str, Any] | None = None

    sf = _session_filters(session_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        while page <= MAX_TRACE_PAGES:
            trace_params: dict[str, Any] = {
                "fromTimestamp": from_ts,
                "toTimestamp": to_ts,
                "orderBy": "timestamp.asc",
                "fields": "core,metrics",
                "page": page,
                "limit": TRACE_PAGE_LIMIT,
            }
            if session_id:
                trace_params["sessionId"] = str(session_id)
            try:
                response = await client.get(
                    f"{base}/api/public/traces",
                    auth=auth,
                    params=trace_params,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                logger.warning("Langfuse traces query failed: %s", exc)
                return {
                    "success": False,
                    "error": "Failed to fetch Langfuse traces.",
                    "details": str(exc),
                }

            data_rows = payload.get("data") or []
            meta = payload.get("meta") or {}
            traces.extend(data_rows)
            pages_fetched += 1
            total_items = _safe_int(meta.get("totalItems"))
            total_pages = _safe_int(meta.get("totalPages"))
            if page >= total_pages or not data_rows:
                break
            page += 1

        totals = {
            "total_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "trace_count": len(traces),
        }
        for tr in traces:
            totals["total_cost_usd"] += _safe_float(tr.get("totalCost"))

        trends: list[dict[str, Any]] = []

        totals_query: dict[str, Any] = {
            "view": "observations",
            "metrics": [
                {"measure": "inputTokens", "aggregation": "sum"},
                {"measure": "outputTokens", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "totalCost", "aggregation": "sum"},
            ],
            "filters": sf,
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "config": {"row_limit": 10},
        }
        try:
            totals_resp = await _langfuse_get_with_retry(
                client,
                url=f"{base}/api/public/metrics",
                auth=auth,
                params={"query": json.dumps(totals_query)},
            )
            totals_metrics_status = totals_resp.status_code
            if totals_resp.status_code < 400:
                totals_data = (totals_resp.json().get("data") or [{}])[0]
                totals_metrics_sample = totals_data if isinstance(totals_data, dict) else None
                totals["input_tokens"] = _metric_int(
                    totals_data, "sum_inputTokens", "inputTokens"
                )
                totals["output_tokens"] = _metric_int(
                    totals_data, "sum_outputTokens", "outputTokens"
                )
                totals["total_tokens"] = _metric_int(
                    totals_data, "sum_totalTokens", "totalTokens"
                )
                metric_cost = _metric_float(totals_data, "sum_totalCost", "totalCost")
                if metric_cost > 0:
                    totals["total_cost_usd"] = metric_cost
        except Exception as exc:
            logger.warning("Langfuse totals metrics query failed (non-blocking): %s", exc)

        trends_query: dict[str, Any] = {
            "view": "observations",
            "metrics": [
                {"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "totalCost", "aggregation": "sum"},
            ],
            "filters": sf,
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "timeDimension": {"granularity": "day"},
            "config": {"row_limit": 400},
        }
        try:
            trends_resp = await _langfuse_get_with_retry(
                client,
                url=f"{base}/api/public/metrics",
                auth=auth,
                params={"query": json.dumps(trends_query)},
            )
            trends_metrics_status = trends_resp.status_code
            if trends_resp.status_code < 400:
                trends_data = trends_resp.json().get("data") or []
                raw_trends = [
                    {
                        "date": str(row.get("time_dimension") or row.get("date") or ""),
                        "total_cost_usd": _metric_float(row, "sum_totalCost", "totalCost"),
                        "total_tokens": _metric_int(row, "sum_totalTokens", "totalTokens"),
                    }
                    for row in trends_data
                    if row.get("time_dimension") or row.get("date")
                ]
                trends = [
                    row
                    for row in raw_trends
                    if _safe_float(row.get("total_cost_usd")) > 0 or _safe_int(row.get("total_tokens")) > 0
                ]
        except Exception as exc:
            logger.warning("Langfuse trends metrics query failed (non-blocking): %s", exc)

        model_rows: list[dict[str, Any]] = []
        metrics_query: dict[str, Any] = {
            "view": "observations",
            "dimensions": [{"field": "providedModelName"}],
            "metrics": [
                {"measure": "count", "aggregation": "count"},
                {"measure": "inputTokens", "aggregation": "sum"},
                {"measure": "outputTokens", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "totalCost", "aggregation": "sum"},
            ],
            "filters": sf,
            "fromTimestamp": from_ts,
            "toTimestamp": to_ts,
            "config": {"row_limit": 50},
        }
        try:
            metrics_resp = await _langfuse_get_with_retry(
                client,
                url=f"{base}/api/public/metrics",
                auth=auth,
                params={"query": json.dumps(metrics_query)},
            )
            models_metrics_status = metrics_resp.status_code
            if metrics_resp.status_code < 400:
                metrics_payload = metrics_resp.json()
                metrics_data = metrics_payload.get("data") or []
                if metrics_data and isinstance(metrics_data[0], dict):
                    models_metrics_sample = metrics_data[0]
                for row in metrics_data:
                    model_name = (
                        row.get("providedModelName")
                        or row.get("model")
                        or row.get("name")
                        or "unknown"
                    )
                    model_rows.append(
                        {
                            "model": str(model_name),
                            "calls": _metric_int(row, "count_count", "count"),
                            "input_tokens": _metric_int(row, "sum_inputTokens", "inputTokens"),
                            "output_tokens": _metric_int(row, "sum_outputTokens", "outputTokens"),
                            "total_tokens": _metric_int(row, "sum_totalTokens", "totalTokens"),
                            "total_cost_usd": _metric_float(row, "sum_totalCost", "totalCost"),
                        }
                    )
        except Exception as exc:
            logger.warning("Langfuse model metrics query failed (non-blocking): %s", exc)

    if (
        _safe_int(totals.get("input_tokens")) == 0
        and _safe_int(totals.get("output_tokens")) == 0
        and _safe_int(totals.get("total_tokens")) == 0
        and model_rows
    ):
        totals["input_tokens"] = sum(_safe_int(r.get("input_tokens")) for r in model_rows)
        totals["output_tokens"] = sum(_safe_int(r.get("output_tokens")) for r in model_rows)
        totals["total_tokens"] = sum(_safe_int(r.get("total_tokens")) for r in model_rows)
        model_cost_total = sum(_safe_float(r.get("total_cost_usd")) for r in model_rows)
        if model_cost_total > 0:
            totals["total_cost_usd"] = model_cost_total

    rate_limited = any(
        status == 429
        for status in (totals_metrics_status, trends_metrics_status, models_metrics_status)
        if status is not None
    )
    used_fallback_snapshot = False
    if (
        rate_limited
        and _safe_int(totals.get("total_tokens")) == 0
        and scope_key in _langfuse_last_success
    ):
        snapshot = _langfuse_last_success[scope_key]
        snapshot_totals = snapshot.get("totals") or {}
        totals["input_tokens"] = _safe_int(snapshot_totals.get("input_tokens"))
        totals["output_tokens"] = _safe_int(snapshot_totals.get("output_tokens"))
        totals["total_tokens"] = _safe_int(snapshot_totals.get("total_tokens"))
        if not model_rows:
            model_rows = snapshot.get("models") or []
        if not trends:
            trends = snapshot.get("trends") or []
        used_fallback_snapshot = totals["total_tokens"] > 0

    model_rows.sort(key=lambda x: x.get("total_cost_usd", 0.0), reverse=True)

    response_payload: dict[str, Any] = {
        "enabled": True,
        "source": "langfuse",
        "scope": scope_key,
        "window": {
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "days": days,
        },
        "totals": totals,
        "trends": trends,
        "models": model_rows,
        "meta": {
            "traces_pages_fetched": pages_fetched,
            "traces_total_items": total_items,
            "traces_collected": len(traces),
            "limited_by_max_pages": pages_fetched >= MAX_TRACE_PAGES,
            "totals_metrics_status": totals_metrics_status,
            "trends_metrics_status": trends_metrics_status,
            "models_metrics_status": models_metrics_status,
            "totals_metrics_sample": totals_metrics_sample,
            "models_metrics_sample": models_metrics_sample,
            "rate_limited": rate_limited,
            "used_fallback_snapshot": used_fallback_snapshot,
            "cached": False,
            "cached_ttl_seconds": LANGFUSE_COSTS_CACHE_TTL_SECONDS,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        },
    }
    _set_langfuse_cached_payload(cache_key, response_payload)
    if _safe_int(totals.get("total_tokens")) > 0:
        _langfuse_last_success[scope_key] = {
            "totals": totals,
            "models": model_rows,
            "trends": trends,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    return {"success": True, "data": response_payload}
