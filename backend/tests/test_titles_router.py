"""Thread title suggestion endpoint."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_titles_suggest_mocked(monkeypatch):
    from app.config import clear_settings_cache
    from app.main import create_app

    clear_settings_cache()
    app = create_app()

    class FakeChoice:
        message = type("M", (), {"content": "  Ayodhya Case Evidence Review  "})()

    class FakeResp:
        choices = [FakeChoice()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    from app.routers import titles as titles_mod

    # Patch the client factory to return object with chat.completions.create = fake_create
    class FakeCompletions:
        async def create(self, **_kwargs):
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(titles_mod, "get_async_openai_client", lambda: FakeClient())
    monkeypatch.setattr(titles_mod, "record_chat_completion_usage", AsyncMock())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/titles/suggest",
            json={
                "user_message": "What did the court hold about the site?",
                "assistant_message": "The court examined evidence and held that...",
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert "title" in data
    assert len(data["title"]) >= 3
    clear_settings_cache()
