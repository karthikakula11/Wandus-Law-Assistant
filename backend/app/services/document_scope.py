"""
Resolve which uploaded document the user means so retrieval can be scoped.

Without this, RAG searches all chunks globally — older uploads can outrank a new file.
"""

from __future__ import annotations

import re
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document

logger = logging.getLogger(__name__)

# e.g. law_test_document.pdf, Act-2024.pdf
_FILENAME_RE = re.compile(
    r"\b[\w.\-]+\.(?:pdf|txt|docx?)\b",
    re.IGNORECASE,
)


def _normalize_title_for_match(title: str) -> list[str]:
    """Variants to search for in user text (longest first)."""
    t = title.strip()
    if not t:
        return []
    out = [t.lower()]
    base = t.rsplit(".", 1)[0].strip()
    if base and base.lower() not in out:
        out.append(base.lower())
    if " " in base:
        out.append(base.lower().replace(" ", "_"))
    return list(dict.fromkeys(out))  # dedupe preserve order


def _text_matches_document(text: str, title: str) -> bool:
    tl = text.lower()
    for needle in sorted(_normalize_title_for_match(title), key=len, reverse=True):
        if len(needle) >= 2 and needle in tl:
            return True
    return False


def _informal_stem_match(text: str, title: str) -> bool:
    """
    Match chat phrasing like "law test doc" to stored title `law_test_document.pdf`:
    require each underscore-separated stem segment to appear; allow "doc" for "document".
    """
    base = title.rsplit(".", 1)[0].strip().lower()
    if not base or ("_" not in base and not any(c.isspace() for c in base)):
        return False
    parts = [p for p in re.split(r"[_\s]+", base) if len(p) >= 2]
    if len(parts) < 2:
        return False
    tl = text.lower()
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        if last and p == "document":
            if "document" in tl or re.search(r"\bdoc\b", tl):
                continue
            return False
        if p in tl:
            continue
        return False
    return True


def _find_doc_in_text(text: str, docs: list[tuple[UUID, str]]) -> UUID | None:
    """Pick one document if `text` clearly references its title (longest title wins)."""
    tl = text
    # Longest titles first to prefer "law_test_document.pdf" over "Act"
    ranked = sorted(docs, key=lambda x: len(x[1]), reverse=True)
    for doc_id, title in ranked:
        if _text_matches_document(tl, title):
            return doc_id
    # Filenames mentioned without matching stored title exactly
    for m in _FILENAME_RE.finditer(tl):
        frag = m.group(0).lower()
        for doc_id, title in ranked:
            if title.lower() == frag or title.rsplit(".", 1)[0].lower() == frag.rsplit(".", 1)[0]:
                return doc_id
    # "law test doc" ↔ law_test_document.pdf
    for doc_id, title in ranked:
        if _informal_stem_match(text, title):
            logger.info("phase=document_scope informal_stem title=%r", title)
            return doc_id
    return None


async def resolve_scope_document_id(
    session: AsyncSession,
    question: str,
    history: list[dict] | None,
    explicit_document_id: UUID | None,
) -> UUID | None:
    """
    If the user (or client) points at one document, return its id for filtered retrieval.

    Priority:
    1. Title/filename in the **current question** (user names a file — wins over UI scope)
    2. **explicit_document_id** from the client API (optional)
    3. Most recent prior **user** message that mentions a document (follow-ups)
    4. Recent **assistant** message that names a file (e.g. prior reply quoted the title)
    """
    result = await session.execute(select(Document.id, Document.title))
    docs = [(row[0], row[1]) for row in result.all()]
    if not docs:
        return None

    q = (question or "").strip()
    if q:
        hit = _find_doc_in_text(q, docs)
        if hit is not None:
            logger.info("phase=document_scope from_question=%s", hit)
            return hit

    if explicit_document_id is not None:
        ok = await session.scalar(
            select(Document.id).where(Document.id == explicit_document_id).limit(1)
        )
        if ok is not None:
            logger.info("phase=document_scope explicit=%s", explicit_document_id)
            return explicit_document_id

    if history:
        for msg in reversed(history):
            if msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            hit = _find_doc_in_text(content, docs)
            if hit is not None:
                logger.info("phase=document_scope from_history_user=%s", hit)
                return hit
        for msg in reversed(history):
            if msg.get("role") != "assistant":
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            hit = _find_doc_in_text(content, docs)
            if hit is not None:
                logger.info("phase=document_scope from_history_assistant=%s", hit)
                return hit

    return None
