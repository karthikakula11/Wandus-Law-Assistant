"""Usage page metadata (Langfuse UI link for tracing). Token/cost totals: ``GET /monitoring/usage/summary``."""

from fastapi import APIRouter

from app.config import get_settings
from app.services.langfuse_tracing import is_langfuse_configured

router = APIRouter(prefix="/usage", tags=["usage"])


def _langfuse_ui_base() -> str:
    s = get_settings()
    h = (s.langfuse_host or "").strip().rstrip("/")
    return h or "https://cloud.langfuse.com"


@router.get("/dashboard")
async def get_usage_dashboard() -> dict:
    """Return Langfuse host + tracing flag for the Usage screen header."""
    configured = is_langfuse_configured()
    base = _langfuse_ui_base()
    return {
        "mode": "langfuse_ui",
        "tracing_configured": configured,
        "langfuse_host": base,
        "open_dashboard_url": base,
        "message": (
            "Totals from llm_usage_log; open Langfuse for traces. Pricing rates from Langfuse GET /api/public/models."
            if configured
            else "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY for tracing and model pricing cache."
        ),
    }
