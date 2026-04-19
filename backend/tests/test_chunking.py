"""Unit tests for structure-aware chunking (no DB)."""

import pytest

from app.chunking import chunk_text


def test_chunk_preserves_case_headers_in_separate_chunks():
    text = (
        "Intro paragraph.\n\n"
        "Case 1: Contract dispute\n\n"
        "Details about contract one. " * 50 + "\n\n"
        "Case 2: Tort\n\n"
        "Details about tort. " * 50
    )
    pairs = chunk_text(text, chunk_size=800, overlap=100)
    joined = "\n".join(p[1] for p in pairs)
    assert "Case 1:" in joined
    assert "Case 2:" in joined
    # At least one chunk should start with or contain a case header
    assert any("Case 1:" in p[1] or "Case 2:" in p[1] for p in pairs)


def test_small_text_single_chunk():
    pairs = chunk_text("Hello world", chunk_size=1200, overlap=200)
    assert len(pairs) == 1
    assert pairs[0][0] == 0
    assert pairs[0][1] == "Hello world"


def test_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_jam_style_word_windows():
    """600-word windows with overlap (Jam with AI TextChunker idea)."""
    words = ["w"] * 700
    text = " ".join(words)
    pairs = chunk_text(
        text,
        strategy="word",
        chunk_words=600,
        overlap_words=100,
        min_chunk_words=10,
    )
    assert len(pairs) >= 2
    assert all(isinstance(p[1], str) and p[1] for p in pairs)


def test_structure_word_keeps_case_headers_in_chunks():
    body = "word " * 400
    text = "Case 1: A\n\n" + body + "\n\nCase 2: B\n\n" + body
    pairs = chunk_text(
        text,
        strategy="structure_word",
        chunk_words=200,
        overlap_words=30,
        min_chunk_words=5,
    )
    joined = "\n".join(p[1] for p in pairs)
    assert "Case 1:" in joined
    assert "Case 2:" in joined
