"""Reciprocal Rank Fusion (shared by retrieval and tests)."""

from __future__ import annotations

from uuid import UUID


def rrf_fuse(
    dense_ids: list[UUID],
    bm25_ids: list[UUID],
    *,
    k: int,
) -> dict[UUID, float]:
    """RRF: higher score is better."""
    scores: dict[UUID, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores
