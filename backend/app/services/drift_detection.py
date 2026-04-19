"""Log retrieval-strength samples and detect distribution drift (TalentVibe-style KS test)."""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import RetrievalDriftSample

logger = logging.getLogger(__name__)

# Cosine distance for normalized vectors is in [0, 2]; map to [0, 1] similarity proxy.
def confidence_from_dense_distance(best_dense_distance: float) -> float:
    d = float(best_dense_distance)
    return max(0.0, min(1.0, 1.0 - d / 2.0))


async def record_retrieval_confidence_sample(
    session: AsyncSession,
    best_dense_distance: float,
) -> None:
    """Persist one confidence point after a document-grounded reply (commits)."""
    try:
        conf = confidence_from_dense_distance(best_dense_distance)
        session.add(RetrievalDriftSample(confidence=conf))
        await session.commit()
    except Exception:
        logger.exception("record_retrieval_confidence_sample failed")
        try:
            await session.rollback()
        except Exception:
            pass


def _finite_float(x: float, *, default: float) -> float:
    """JSON cannot encode NaN/inf; scipy may return NaN in edge cases."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


async def detect_drift(
    session: AsyncSession,
    *,
    days: int = 30,
    threshold: float = 0.1,
    min_samples: int | None = None,
    split_min_total: int | None = None,
) -> dict[str, Any] | None:
    """
    Compare confidence distributions: current window vs previous window.
    Uses Kolmogorov–Smirnov; optional split-half if little history.
    """
    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy unavailable — drift detection disabled")
        return None

    s = get_settings()
    ms = min_samples if min_samples is not None else s.drift_min_samples
    st = split_min_total if split_min_total is not None else s.drift_split_min_total
    if st < 2 * ms:
        st = 2 * ms

    try:
        return await _detect_drift_inner(
            session,
            stats,
            days=days,
            threshold=threshold,
            min_samples=ms,
            split_min_total=st,
        )
    except Exception:
        logger.exception("detect_drift failed")
        return None


async def _detect_drift_inner(
    session: AsyncSession,
    stats: Any,
    *,
    days: int,
    threshold: float,
    min_samples: int,
    split_min_total: int,
) -> dict[str, Any] | None:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    prev_start_date = start_date - timedelta(days=days)

    cur_stmt = select(RetrievalDriftSample.confidence, RetrievalDriftSample.created_at).where(
        RetrievalDriftSample.created_at >= start_date,
        RetrievalDriftSample.created_at <= end_date,
    )
    prev_stmt = select(RetrievalDriftSample.confidence, RetrievalDriftSample.created_at).where(
        RetrievalDriftSample.created_at >= prev_start_date,
        RetrievalDriftSample.created_at < start_date,
    )

    cur_rows = (await session.execute(cur_stmt)).all()
    prev_rows = (await session.execute(prev_stmt)).all()

    current_preds = list(cur_rows)
    prev_preds = list(prev_rows)

    use_split_comparison = False
    if len(prev_preds) < min_samples and len(current_preds) >= split_min_total:
        sorted_current = sorted(current_preds, key=lambda p: p[1])
        mid = len(sorted_current) // 2
        prev_preds = sorted_current[:mid]
        current_preds = sorted_current[mid:]
        use_split_comparison = True

    if len(current_preds) < min_samples or len(prev_preds) < min_samples:
        return None

    current_confidences = [float(p[0]) for p in current_preds]
    prev_confidences = [float(p[0]) for p in prev_preds]

    if len(current_confidences) >= min_samples and len(prev_confidences) >= min_samples:
        ks_statistic, p_value = stats.ks_2samp(current_confidences, prev_confidences)
        p_value = _finite_float(p_value, default=1.0)
        ks_statistic = _finite_float(ks_statistic, default=0.0)
        drift_detected = bool(p_value < threshold)
        cur_m = float(statistics.mean(current_confidences))
        prev_m = float(statistics.mean(prev_confidences))
        return {
            "drift_detected": drift_detected,
            "p_value": p_value,
            "ks_statistic": ks_statistic,
            "threshold": threshold,
            "detection_method": (
                "confidence_distribution" if not use_split_comparison else "confidence_distribution_split"
            ),
            "current_period_count": len(current_preds),
            "previous_period_count": len(prev_preds),
            "current_avg_confidence": cur_m,
            "previous_avg_confidence": prev_m,
            "confidence_shift": cur_m - prev_m,
            "confidence_coverage": {
                "current": 1.0,
                "previous": 1.0,
            },
            "comparison_type": "split_period" if use_split_comparison else "historical_period",
            "message": (
                "Comparing first half vs second half of current period"
                if use_split_comparison
                else None
            ),
            "window_days": days,
        }

    return None
