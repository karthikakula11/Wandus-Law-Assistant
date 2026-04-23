import logging
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

from app.config import clear_settings_cache, get_settings

logger = logging.getLogger(__name__)

_REPO_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
_last_env_mtime: float | None = None


async def check_db_ready() -> bool:
    """Ping Postgres and verify pgvector extension exists."""
    global _last_env_mtime
    if _REPO_ROOT_ENV.is_file():
        m = _REPO_ROOT_ENV.stat().st_mtime
        if _last_env_mtime is None or m > _last_env_mtime:
            _last_env_mtime = m
            load_dotenv(_REPO_ROOT_ENV, override=True)
            clear_settings_cache()
            from app.database import dispose_db

            await dispose_db()

    settings = get_settings()
    conn = None
    try:
        conn = await asyncpg.connect(settings.database_url, ssl=False)
        await conn.execute("SELECT 1")
        row = await conn.fetchrow(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1"
        )
        return row is not None
    except Exception as e:
        logger.warning("db_ready_failed: %s", e)
        return False
    finally:
        if conn is not None:
            await conn.close()
