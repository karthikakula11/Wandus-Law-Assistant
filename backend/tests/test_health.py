import pytest
from httpx import ASGITransport, AsyncClient

from app.config import clear_settings_cache
from app.main import create_app


@pytest.mark.asyncio
async def test_health_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_returns_503_when_db_unavailable(monkeypatch):
    async def fail_ready():
        return False

    import app.main as main_mod

    monkeypatch.setattr(main_mod, "check_db_ready", fail_ready)

    clear_settings_cache()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/ready")
    assert r.status_code == 503
    assert r.json()["detail"] == "not_ready"
    clear_settings_cache()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ready_ok_when_database_up():
    """Requires Docker Postgres and DATABASE_URL in environment (run from repo root with .env)."""
    import os

    if not os.environ.get("DATABASE_URL") or "59999" in os.environ.get("DATABASE_URL", ""):
        pytest.skip("Set DATABASE_URL to a running Postgres for integration test")

    clear_settings_cache()
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}
    clear_settings_cache()
