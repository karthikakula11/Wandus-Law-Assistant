"""
Ingest: document → chunks → embeddings → DB (+ optional OpenSearch).

**Jam with AI (cloned) reference** — indexing flow receives **extracted** full text, then chunks;
see ``reference/production-agentic-rag-course/src/services/indexing/hybrid_indexer.py`` and
``text_chunker.py`` (``chunk_paper`` / ``chunk_text``). Our order matches: ``bytes_to_document_text``
/ ``chunk_text`` / ``embed_texts`` / persist — see ``pdf_extract.py`` for the Docling → PyMuPDF note.
"""

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking import chunk_text
from app.config import get_settings
from app.embeddings import embed_texts
from app.models import Chunk, Document
from app.services import search_index

logger = logging.getLogger(__name__)


async def ingest_document(
    session: AsyncSession,
    *,
    title: str,
    text: str,
    source_uri: str | None,
) -> tuple[uuid.UUID, int]:
    doc = Document(title=title, source_uri=source_uri)
    session.add(doc)
    await session.flush()

    s = get_settings()
    pairs = chunk_text(
        text,
        strategy=s.chunking_strategy,
        chunk_size=s.chunk_size_chars,
        overlap=s.chunk_overlap_chars,
        chunk_words=s.chunk_words,
        overlap_words=s.chunk_overlap_words,
        min_chunk_words=s.chunk_min_words,
    )
    if not pairs:
        await session.commit()
        return doc.id, 0

    contents = [p[1] for p in pairs]
    embeddings = await embed_texts(contents)

    chunks_added: list[Chunk] = []
    for (chunk_index, content), emb in zip(pairs, embeddings, strict=True):
        ch = Chunk(
            document_id=doc.id,
            chunk_index=chunk_index,
            content=content,
            meta=None,
            embedding=emb,
        )
        session.add(ch)
        chunks_added.append(ch)

    await session.flush()
    await session.commit()

    if (s.opensearch_url or "").strip():
        await search_index.ensure_index_exists()
        for ch in chunks_added:
            emb: list[float] | None = None
            if s.opensearch_unified_hybrid:
                ev = ch.embedding
                if hasattr(ev, "tolist"):
                    emb = ev.tolist()  # type: ignore[assignment]
                else:
                    emb = [float(x) for x in ev]
            ok = await search_index.index_chunk(
                chunk_id=ch.id,
                document_id=doc.id,
                document_title=title,
                content=ch.content,
                embedding=emb,
            )
            if not ok:
                logger.warning("OpenSearch index failed for chunk %s", ch.id)

    return doc.id, len(pairs)
