from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings

_engine = None
_session_factory = None


def _async_database_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            _async_database_url(settings.database_url),
            echo=False,
            poolclass=NullPool,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def dispose_db() -> None:
    """Close the engine (e.g. between pytest async tests using different event loops)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        yield session
