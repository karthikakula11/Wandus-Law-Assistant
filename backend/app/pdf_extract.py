"""
Extract plain text from uploads.

**Jam with AI (cloned) reference**

- PDFs are parsed with **Docling** in the course repo — ``pdf_parser/parser.py``,
  ``pdf_parser/docling.py`` — producing clean text before chunking.

**This app**

- **Docling** runs first up to ``DOCLING_MAX_PAGES`` (e.g. 400). If the PDF has **more**
  pages, **PyMuPDF** extracts **only** pages after that range so the full document is
  covered without duplicating the Docling window.
- **PyMuPDF** dict/span extraction; we reject outputs that still look like raw PDF syntax.
  Magic bytes are checked **before** filename so a mis-labeled ``.txt`` upload is never
  UTF-8–decoded as raw PDF bytes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# xref table lines: "003086 00000 n"
_XREF_LINE = re.compile(r"^\s*\d{4,6}\s+\d{4,6}\s+n\s*$", re.MULTILINE)


def _looks_like_pdf(data: bytes) -> bool:
    if len(data) < 4 or data[:4] != b"%PDF":
        return False
    return len(data) < 5 or data[4:5] == b"-"


def _normalize_pdf_text(s: str) -> str:
    s = s.replace("\x00", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()


def _looks_like_raw_pdf_syntax(s: str) -> bool:
    """True if this still looks like PDF file structure, not reading text."""
    if not s or len(s) < 8:
        return False
    t = s.lstrip()
    if t.startswith("%PDF"):
        return True
    head = t[:8000]
    tail = s[-2000:] if len(s) > 2000 else s
    if head.count("endobj") >= 4 and head.count("<<") >= 3:
        return True
    if "startxref" in head and "%%EOF" in tail:
        return True
    if "trailer" in head[:2000] and "/Root" in head and "/Size" in head:
        return True
    # Many xref-style lines = not body text
    if len(_XREF_LINE.findall(s[:5000])) >= 4:
        return True
    return False


def _text_letter_ratio(sample: str) -> float:
    """Higher means more human-readable prose; raw/binary PDF pulls this down."""
    if not sample:
        return 0.0
    chunk = sample[:12000]
    good = sum(1 for c in chunk if c.isalpha() or c.isspace() or c in ".,;:!?-'\"()[]{}")
    return good / max(len(chunk), 1)


def _extract_text_dict_mode(
    doc: Any,
    start: int = 0,
    end: int | None = None,
) -> str:
    """Text spans only (block type 0). Optional page range ``[start, end)`` (0-based)."""
    import fitz

    last = doc.page_count if end is None else min(end, doc.page_count)
    page_texts: list[str] = []
    for i in range(max(0, start), last):
        page = doc[i]
        page_lines: list[str] = []
        d = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                piece = "".join(s.get("text", "") for s in spans)
                if piece.strip():
                    page_lines.append(piece)
        if page_lines:
            page_texts.append("\n".join(page_lines))
    return "\n\n".join(page_texts)


def _extract_text_get_text_sorted(
    doc: Any,
    start: int = 0,
    end: int | None = None,
) -> str:
    """``get_text(sort=True)`` per page; optional range ``[start, end)``."""
    last = doc.page_count if end is None else min(end, doc.page_count)
    parts: list[str] = []
    for i in range(max(0, start), last):
        page = doc[i]
        t = page.get_text(sort=True)
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _extract_text_page_range(doc: Any, start: int, end: int) -> str:
    """Best-effort text for pages ``[start, end)`` (0-based)."""
    if start >= end or start >= doc.page_count:
        return ""
    dict_text = _extract_text_dict_mode(doc, start, end)
    plain_text = _extract_text_get_text_sorted(doc, start, end)
    return _pick_best_extraction(dict_text, plain_text)


def _acceptable_body_text(t: str, label: str) -> bool:
    if not t.strip():
        return False
    if _looks_like_raw_pdf_syntax(t):
        logger.warning("pdf_extract: reject %s — still looks like PDF syntax / xref / trailer", label)
        return False
    lr = _text_letter_ratio(t)
    # Short snippets can be all-caps headings; long noise (streams) sinks letter ratio
    if len(t) > 120 and lr < 0.14:
        logger.warning("pdf_extract: reject %s — low letter ratio %.2f (likely binary stream)", label, lr)
        return False
    return True


def _pick_best_extraction(dict_text: str, plain_text: str) -> str:
    """
    Prefer **dict** spans (real text blocks). Use ``get_text`` only if dict is empty/unusable.
    """
    d = dict_text.strip()
    if d and _acceptable_body_text(d, "dict"):
        return d

    p = plain_text.strip()
    if p and _acceptable_body_text(p, "plain"):
        return p

    logger.error(
        "pdf_extract: no usable text layer (course uses Docling). "
        "Re-export PDF or use OCR for scanned pages."
    )
    return ""


def extract_text_from_pdf(data: bytes) -> str:
    """
    Docling for the first ``docling_max_pages`` pages when enabled; if the PDF is longer,
    append PyMuPDF text for the remaining pages. If Docling fails, fall back to full PyMuPDF.
    """
    from app.config import get_settings

    settings = get_settings()
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page_count = doc.page_count
        max_docling = settings.docling_max_pages
        pymupdf_after_docling_miss = False

        if settings.use_docling_for_pdf:
            from app.services.docling_pdf import extract_text_with_docling

            docling_text = extract_text_with_docling(
                data,
                max_pages=max_docling,
            )
            if docling_text and len(docling_text.strip()) >= 10:
                if page_count <= max_docling:
                    return _normalize_pdf_text(docling_text)

                tail = _extract_text_page_range(doc, max_docling, page_count)
                merged = docling_text.rstrip()
                if tail.strip():
                    merged = merged + "\n\n" + tail
                    logger.info(
                        "pdf_extract: docling pages 1-%s + pymupdf pages %s-%s",
                        max_docling,
                        max_docling + 1,
                        page_count,
                    )
                else:
                    logger.warning(
                        "pdf_extract: docling ok for first %s pages but no tail text "
                        "(pages %s-%s); using docling head only",
                        max_docling,
                        max_docling + 1,
                        page_count,
                    )
                return _normalize_pdf_text(merged)

            pymupdf_after_docling_miss = True
            logger.info(
                "pdf_extract: Running PyMuPDF on full document (%s pages) after Docling "
                "did not return usable text.",
                page_count,
            )

        dict_text = _extract_text_dict_mode(doc)
        plain_text = _extract_text_get_text_sorted(doc)
        chosen = _pick_best_extraction(dict_text, plain_text)
        if pymupdf_after_docling_miss:
            if chosen.strip():
                logger.info(
                    "pdf_extract: PyMuPDF fallback extracted %s characters.",
                    len(chosen.strip()),
                )
            else:
                logger.error(
                    "pdf_extract: PyMuPDF also found no usable text layer (e.g. scanned "
                    "PDF without OCR). Try another export or paste text."
                )
        return _normalize_pdf_text(chosen)
    finally:
        doc.close()


def bytes_to_document_text(filename: str | None, data: bytes) -> str:
    """
    **Important:** PDF is detected by **magic bytes first**, not only the filename.
    Otherwise a ``.txt`` upload containing raw PDF bytes is UTF-8–decoded and chunked as garbage
    (this matches the bad chunks you see: ``%PDF-1.4``, xref, ``endstream``, etc.).
    """
    name = (filename or "").lower()

    if _looks_like_pdf(data):
        return extract_text_from_pdf(data)

    if name.endswith(".pdf"):
        if len(data) >= 4 and data[:4] != b"%PDF":
            logger.warning("File named .pdf but missing %%PDF magic; attempting extract anyway")
        return extract_text_from_pdf(data)

    raw = data.decode("utf-8", errors="replace")
    return _normalize_pdf_text(raw)
