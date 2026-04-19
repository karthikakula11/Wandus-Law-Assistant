"""Tests for usage recording orchestrator health report."""

from unittest.mock import AsyncMock, patch

import pytest

from app.monitoring.usage_recording_health import build_usage_recording_health


class _FakeResult:
    def __init__(self, *, scalar=None, rows=None):
        self._scalar = scalar
        self._rows = rows or []

    def scalar(self):
        return self._scalar

    def all(self):
        return self._rows


def _session_with_execute_chain(chain: list[_FakeResult]):
    session = AsyncMock()
    it = iter(chain)

    async def _exec(*_a, **_kw):
        return next(it)

    session.execute = AsyncMock(side_effect=_exec)
    return session


def _pricing_ok(models: int = 3):
    return {
        "source": "langfuse_get_api_public_models",
        "models_cached": models,
        "langfuse_configured": True,
        "cost_formula": "tokens_in * inputPrice + tokens_out * outputPrice (USD per token)",
        "last_refresh_at": None,
        "last_refresh_error": None,
    }


@pytest.mark.asyncio
async def test_build_usage_recording_health_ok_with_usage():
    chain = [
        _FakeResult(scalar=100),
        _FakeResult(scalar=12),
        _FakeResult(scalar=3),
        _FakeResult(scalar=0),
        _FakeResult(rows=[]),
        _FakeResult(rows=[("chat", 8)]),
    ]
    session = _session_with_execute_chain(chain)
    with patch(
        "app.monitoring.usage_recording_health.llm_pricing.get_pricing_status",
        return_value=_pricing_ok(3),
    ):
        out = await build_usage_recording_health(session)
    assert out["service"] == "usage_recording_orchestrator"
    assert out["status"] == "ok"
    assert out["checks"]["langfuse_keys_configured"] is True
    assert out["checks"]["pricing_catalog_has_models"] is True
    assert out["checks"]["no_recent_null_cost_rows"] is True
    assert out["llm_usage_log"]["total_rows"] == 100
    assert out["llm_usage_log"]["rows_last_7d"] == 12
    assert any("healthy" in r.lower() for r in out["recommendations"])


@pytest.mark.asyncio
async def test_build_usage_recording_health_null_cost_degraded():
    chain = [
        _FakeResult(scalar=50),
        _FakeResult(scalar=10),
        _FakeResult(scalar=2),
        _FakeResult(scalar=3),
        _FakeResult(rows=[("gpt-4o-mini", 3)]),
        _FakeResult(rows=[]),
    ]
    session = _session_with_execute_chain(chain)
    with patch(
        "app.monitoring.usage_recording_health.llm_pricing.get_pricing_status",
        return_value=_pricing_ok(2),
    ):
        out = await build_usage_recording_health(session)
    assert out["status"] == "degraded"
    assert out["checks"]["no_recent_null_cost_rows"] is False
    assert out["llm_usage_log"]["null_cost_rows_last_7d"] == 3
    assert any("NULL" in r or "null" in r for r in out["recommendations"])


@pytest.mark.asyncio
async def test_build_usage_recording_health_pricing_refresh_error_degraded():
    chain = [
        _FakeResult(scalar=10),
        _FakeResult(scalar=2),
        _FakeResult(scalar=1),
        _FakeResult(scalar=0),
        _FakeResult(rows=[]),
        _FakeResult(rows=[]),
    ]
    session = _session_with_execute_chain(chain)
    p = _pricing_ok(2)
    p["last_refresh_error"] = "connection refused"
    with patch(
        "app.monitoring.usage_recording_health.llm_pricing.get_pricing_status",
        return_value=p,
    ):
        out = await build_usage_recording_health(session)
    assert out["status"] == "degraded"
    assert any("refresh" in r.lower() for r in out["recommendations"])
