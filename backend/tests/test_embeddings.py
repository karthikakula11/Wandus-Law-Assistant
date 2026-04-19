"""Embedding batching and token limits."""

from app.embeddings import (
    _batch_pieces,
    _expand_for_api,
    _mean_embedding,
)


def test_mean_embedding_single():
    assert _mean_embedding([[1.0, 2.0]]) == [1.0, 2.0]


def test_mean_embedding_average():
    assert _mean_embedding([[0.0, 2.0], [2.0, 0.0]]) == [1.0, 1.0]


def test_expand_splits_oversized(monkeypatch):
    from app import embeddings as emb_mod

    def fake_split(t: str):
        if len(t) > 100:
            return [t[:50], t[50:100], t[100:150]]
        return [t]

    monkeypatch.setattr(emb_mod, "_split_oversized_inputs", fake_split)
    texts = ["x" * 200]
    orig, pieces = _expand_for_api(texts)
    assert len(pieces) == 3
    assert orig == [0, 0, 0]


def test_batch_pieces_splits_large_corpus():
    # Many small chunks should form multiple API batches
    chunks = ["x" * 1200 for _ in range(2000)]
    orig, pieces = _expand_for_api(chunks)
    assert len(orig) == len(pieces) == 2000
    batches = _batch_pieces(pieces)
    assert len(batches) >= 2
    assert sum(len(b) for b in batches) == len(pieces)


