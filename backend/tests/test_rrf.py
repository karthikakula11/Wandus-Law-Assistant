"""Unit tests for RRF fusion (no DB)."""

from uuid import uuid4

from app.services.rrf_util import rrf_fuse


def test_rrf_prefers_item_in_both_lists():
    a, b, c = uuid4(), uuid4(), uuid4()
    dense = [a, b]
    bm25 = [b, c]
    scores = rrf_fuse(dense, bm25, k=60)
    assert scores[b] > scores[a]
    assert scores[b] > scores[c]


def test_rrf_empty_legs():
    x = uuid4()
    assert rrf_fuse([], [], k=60) == {}
    assert rrf_fuse([x], [], k=60)[x] > 0
