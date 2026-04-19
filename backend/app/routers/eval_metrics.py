"""Serve latest offline benchmark results (``eval/results/latest.json``) and retrieval drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session
from app.services.drift_detection import detect_drift

router = APIRouter(prefix="/eval", tags=["eval"])


def _results_path() -> Path:
    return Path(__file__).resolve().parents[2] / "eval" / "results" / "latest.json"


@router.get("/summary")
async def get_eval_summary() -> dict[str, Any]:
    path = _results_path()
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="No eval results yet. Run: python scripts/run_eval.py (from backend/)",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path}: {e}") from e


@router.get("/drift")
async def get_retrieval_drift(
    session: AsyncSession = Depends(get_session),
    days: int = Query(7, ge=1, le=365),
    threshold: float = Query(0.1, ge=0.001, le=0.5),
) -> dict[str, Any]:
    """
    Kolmogorov–Smirnov test on live retrieval confidence (TalentVibe-style; no gold labels).
    Compares last ``days`` vs the prior ``days``; needs enough samples per side (configurable),
    or splits the current window when history is short.
    """
    try:
        result = await detect_drift(session, days=days, threshold=threshold)
    except ProgrammingError as e:
        if "retrieval_drift_samples" in str(e).lower() or "undefinedtable" in str(e).lower():
            msg = (
                "Database table missing. From backend/ run: "
                "`.venv/bin/alembic upgrade head`"
            )
        else:
            msg = f"Database error: {e}"
        return {
            "success": True,
            "data": {
                "drift_detected": False,
                "message": msg,
                "threshold": threshold,
                "window_days": days,
            },
        }
    except Exception as e:
        return {
            "success": True,
            "data": {
                "drift_detected": False,
                "message": f"Drift check failed ({type(e).__name__}): {e}",
                "threshold": threshold,
                "window_days": days,
            },
        }

    if result is None:
        s = get_settings()
        return {
            "success": True,
            "data": {
                "drift_detected": False,
                "message": (
                    f"Insufficient data: need at least {s.drift_min_samples} samples in each window "
                    f"(document-grounded chats), or run `python scripts/seed_drift_demo.py stable|shift` for a demo."
                ),
                "threshold": threshold,
                "window_days": days,
            },
        }
    return {"success": True, "data": result}
