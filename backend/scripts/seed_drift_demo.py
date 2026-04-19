#!/usr/bin/env python3
"""
Insert live-style retrieval confidence samples into ``retrieval_drift_samples`` for demos.

Same table as real chat logging — use only for presentations. After seeding, open Drift in the UI
with default 7-day windows.

Usage (from ``backend/``):
  .venv/bin/python scripts/seed_drift_demo.py stable   # expect **No drift** badge
  .venv/bin/python scripts/seed_drift_demo.py shift    # expect **Drift detected** badge
  .venv/bin/python scripts/seed_drift_demo.py stable --clear   # wipe table first
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _spread_times(start: datetime, end: datetime, n: int) -> list[datetime]:
    dur = (end - start).total_seconds()
    return [start + timedelta(seconds=dur * (i + 0.5) / n) for i in range(n)]


async def _run(mode: str, clear: bool) -> None:
    from sqlalchemy import delete

    from app.database import get_session_factory
    from app.models import RetrievalDriftSample

    factory = get_session_factory()
    async with factory() as session:
        if clear:
            await session.execute(delete(RetrievalDriftSample))
            await session.commit()

        now = datetime.now(timezone.utc)
        cur_start = now - timedelta(days=7)
        prev_start = now - timedelta(days=14)
        prev_end = cur_start

        n = 8
        prev_times = _spread_times(prev_start, prev_end, n)
        cur_times = _spread_times(cur_start, now, n)

        if mode == "stable":
            # Same distribution in both windows → KS p≈1, **No drift** badge
            base = [0.68, 0.72, 0.75, 0.76, 0.74, 0.73, 0.77, 0.71]
            prev_conf = list(base)
            cur_conf = list(base)
        else:
            # Clear separation → **Drift detected** badge
            prev_conf = [0.38 + i * 0.04 for i in range(n)]
            cur_conf = [0.88 + (i % 3) * 0.02 for i in range(n)]

        for t, c in zip(prev_times, prev_conf, strict=True):
            session.add(
                RetrievalDriftSample(id=uuid.uuid4(), created_at=t, confidence=min(1.0, max(0.0, c)))
            )
        for t, c in zip(cur_times, cur_conf, strict=True):
            session.add(
                RetrievalDriftSample(id=uuid.uuid4(), created_at=t, confidence=min(1.0, max(0.0, c)))
            )
        await session.commit()
        print(f"Inserted {2 * n} demo rows ({mode}). Open Drift UI (7-day window) and Refresh.")


def main() -> None:
    p = argparse.ArgumentParser(description="Seed retrieval drift samples for demo")
    p.add_argument(
        "mode",
        choices=("stable", "shift"),
        help="stable ≈ no drift; shift ≈ drift detected",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Delete all rows in retrieval_drift_samples before inserting",
    )
    args = p.parse_args()
    asyncio.run(_run(args.mode, args.clear))


if __name__ == "__main__":
    main()
