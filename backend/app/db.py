import asyncpg

from app.config import get_settings


async def check_db_ready() -> bool:
    """Ping Postgres and verify pgvector extension exists."""
    settings = get_settings()
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute("SELECT 1")
        row = await conn.fetchrow(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1"
        )
        return row is not None
    finally:
        await conn.close()
