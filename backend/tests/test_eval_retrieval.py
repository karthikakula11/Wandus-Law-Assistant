"""Unit tests for eval retrieval helpers (no DB)."""

from uuid import UUID

import pytest

from app.services.eval_retrieval import mean_metric, recall_hit


def test_recall_hit():
    g1 = UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11")
    g2 = UUID("b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a12")
    assert recall_hit([g1], [g1, g2]) is True
    assert recall_hit([g1], [g2]) is False
    assert recall_hit([], [g1]) is False
    assert recall_hit([g1], [g2, g1], k=1) is False
    assert recall_hit([g1], [g2, g1], k=2) is True


def test_mean_metric():
    assert mean_metric([True, False, True]) == pytest.approx(2 / 3)
    assert mean_metric([]) is None
