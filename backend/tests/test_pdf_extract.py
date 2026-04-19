"""PDF routing and rejection of raw-PDF-as-text (no PyMuPDF file required)."""

import pytest


def test_magic_bytes_force_pdf_path_not_utf8_decode(monkeypatch):
    """Misnamed .txt must not UTF-8 decode raw PDF bytes (source of garbage chunks)."""
    called = {}

    def fake_extract(data: bytes) -> str:
        called["yes"] = True
        return "extracted legal text only"

    monkeypatch.setattr("app.pdf_extract.extract_text_from_pdf", fake_extract)

    from app.pdf_extract import bytes_to_document_text

    # Minimal PDF header (same magic browsers use)
    pdfish = b"%PDF-1.4\n%fake rest of file would be binary"
    out = bytes_to_document_text("notes.txt", pdfish)
    assert called.get("yes") is True
    assert out == "extracted legal text only"


def test_plain_txt_still_decoded():
    from app.pdf_extract import bytes_to_document_text

    out = bytes_to_document_text("a.txt", b"Section 1. Short title.\n")
    assert "Short title" in out


def test_reject_raw_pdf_syntax():
    from app.pdf_extract import _looks_like_raw_pdf_syntax

    assert _looks_like_raw_pdf_syntax("%PDF-1.4\nfoo") is True
    assert _looks_like_raw_pdf_syntax("The parties agree that rent is due.") is False
