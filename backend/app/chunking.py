"""
Chunking strategies.

**Jam with AI (cloned) reference** — ``reference/production-agentic-rag-course/src/services/indexing/text_chunker.py``:
``TextChunker`` uses word-based windows (default **600** words, **100** overlap),
``re.findall(r"\\S+", ...)`` for tokens, and optional section-based ``chunk_paper`` when
structured sections exist.

**This app**: ``strategy=word`` / ``structure_word`` + env ``CHUNK_WORDS`` / ``CHUNK_OVERLAP_WORDS``
mirrors those defaults; ``structure`` keeps legal-PDF-oriented heading/case splits then char windows.
Input text must already be clean plain text (from ``pdf_extract`` / ingest) — same contract as the
course pipeline before ``TextChunker.chunk_text``.

- **structure** (default): split at headings / Case markers, then character windows (legacy).
- **word**: Jam with AI–style sliding windows over words (default 600 words, 100 overlap).
- **structure_word**: structure splits first, then word windows within each segment (best for law PDFs + course-like sizes).
"""

from __future__ import annotations

import re
from typing import Literal

# New section: markdown heading, "Case N:", or horizontal rule line
_SECTION_BREAK = re.compile(
    r"(?=(?:^|\n)\s*(?:"
    r"#{1,6}\s+"  # markdown headings
    r"|Case\s*\d+\s*[:.)]"  # Case 1: / Case 2)
    r"|\*{3,}"  # ***
    r"|---+"
    r"))",
    re.MULTILINE,
)


def _segment_by_structure(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SECTION_BREAK.split(text) if p.strip()]
    if len(parts) <= 1:
        return [text] if text else []
    return parts


def _sliding_pieces_char(text: str, chunk_size: int, overlap: int) -> list[str]:
    out: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = end - overlap
    return out


def _split_words(text: str) -> list[str]:
    """Same tokenization idea as Jam with AI TextChunker (`re.findall(r'\\S+', ...)`)."""
    return re.findall(r"\S+", text)


def _sliding_pieces_word(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[str]:
    """
    Jam with AI–style: overlapping word windows (default 600 / 100).
    Short texts below `min_chunk_words` become a single chunk.
    """
    words = _split_words(text)
    if not words:
        return []
    if len(words) < min_chunk_words:
        return [" ".join(words)]

    if overlap_words >= chunk_words:
        overlap_words = max(0, chunk_words // 6)

    out: list[str] = []
    current_position = 0
    while current_position < len(words):
        chunk_end = min(current_position + chunk_words, len(words))
        piece = words[current_position:chunk_end]
        out.append(" ".join(piece))
        if chunk_end >= len(words):
            break
        current_position += chunk_words - overlap_words
    return out


def _chunk_structure_char(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, str]]:
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    segments = _segment_by_structure(text)
    out: list[tuple[int, str]] = []
    idx = 0

    for seg in segments:
        if len(seg) <= chunk_size:
            out.append((idx, seg))
            idx += 1
            continue
        for piece in _sliding_pieces_char(seg, chunk_size, overlap):
            out.append((idx, piece))
            idx += 1

    return out


def _chunk_word_only(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[tuple[int, str]]:
    pieces = _sliding_pieces_word(text.strip(), chunk_words, overlap_words, min_chunk_words)
    return [(i, p) for i, p in enumerate(pieces)]


def _chunk_structure_word(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> list[tuple[int, str]]:
    """Headings / Case boundaries preserved, then Jam-style word windows per segment."""
    text = text.strip()
    if not text:
        return []

    segments = _segment_by_structure(text)
    out: list[tuple[int, str]] = []
    idx = 0
    for seg in segments:
        pieces = _sliding_pieces_word(seg, chunk_words, overlap_words, min_chunk_words)
        for p in pieces:
            out.append((idx, p))
            idx += 1
    return out


def chunk_text(
    text: str,
    *,
    strategy: Literal["structure", "word", "structure_word"] = "structure",
    chunk_size: int = 1200,
    overlap: int = 200,
    chunk_words: int = 600,
    overlap_words: int = 100,
    min_chunk_words: int = 100,
) -> list[tuple[int, str]]:
    """
    Return (chunk_index, content) pairs.

    **strategy**
    - ``structure`` — split at ``##`` / ``Case N:`` / rules, then **character** windows.
    - ``word`` — **word** windows only (Jam with AI course defaults when using 600/100).
    - ``structure_word`` — structure splits, then **word** windows (recommended for legal PDFs).
    """
    text = text.strip()
    if not text:
        return []

    if strategy == "word":
        return _chunk_word_only(text, chunk_words, overlap_words, min_chunk_words)
    if strategy == "structure_word":
        return _chunk_structure_word(text, chunk_words, overlap_words, min_chunk_words)
    return _chunk_structure_char(text, chunk_size, overlap)
