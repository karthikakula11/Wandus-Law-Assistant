"""Unit tests for drift confidence mapping (no DB)."""

from app.services.drift_detection import confidence_from_dense_distance


def test_confidence_from_dense_distance():
    assert confidence_from_dense_distance(0.0) == 1.0
    assert confidence_from_dense_distance(2.0) == 0.0
    assert confidence_from_dense_distance(1.0) == 0.5
    assert 0 <= confidence_from_dense_distance(0.5) <= 1
