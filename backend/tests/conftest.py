import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

_root = Path(__file__).resolve().parents[2]
load_dotenv(_root / ".env")

# Default test env if missing (invalid URL — tests mock DB unless integration)
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:59999/none",
)
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")


@pytest.fixture
async def client():
    from app.config import clear_settings_cache
    from app.main import create_app

    clear_settings_cache()
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    clear_settings_cache()
