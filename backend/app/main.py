import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import get_settings
from app.db import check_db_ready
from app.routers import agent_graph, chat, documents, eval_metrics, ingest, jam_api, memory, monitoring, titles, usage_dashboard, user_chat_state, users
from app.services import search_index

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.langfuse_tracing import init_langfuse_client_from_settings

    init_langfuse_client_from_settings()
    settings = get_settings()
    if (settings.opensearch_url or "").strip():
        try:
            ok = await search_index.ensure_index_exists()
            logger.info("opensearch_index_ready=%s", ok)
        except httpx.HTTPError as e:
            logger.warning(
                "OpenSearch unreachable at startup (index not ensured): %s",
                e,
            )
    pricing_task: asyncio.Task | None = None
    try:
        from app.monitoring import llm_pricing

        await llm_pricing.refresh_from_langfuse()
        pricing_task = asyncio.create_task(llm_pricing.pricing_refresh_loop())
    except Exception as e:
        logger.debug("llm_pricing startup skipped: %s", e)

    yield

    if pricing_task is not None:
        pricing_task.cancel()
        try:
            await pricing_task
        except asyncio.CancelledError:
            pass
    try:
        from app.services.langfuse_tracing import flush_langfuse

        flush_langfuse()
    except Exception:
        logger.debug("langfuse shutdown flush skipped", exc_info=True)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


def create_app() -> FastAPI:
    get_settings()
    app = FastAPI(
        title="Pintu Law RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        ok = await check_db_ready()
        if not ok:
            raise HTTPException(status_code=503, detail="not_ready")
        return {"status": "ready"}

    app.include_router(ingest.router)
    app.include_router(documents.router)
    app.include_router(agent_graph.router)
    app.include_router(chat.router)
    app.include_router(memory.router)
    app.include_router(users.router)
    app.include_router(user_chat_state.router)
    app.include_router(eval_metrics.router)
    app.include_router(titles.router)
    app.include_router(usage_dashboard.router)
    app.include_router(monitoring.router, prefix="/monitoring")
    app.include_router(monitoring.router, prefix="/api/v1/monitoring")
    # Jam with AI course–compatible API ( /ask, /stream, /hybrid-search )
    app.include_router(jam_api.router, prefix="/api/v1")

    return app


app = create_app()
