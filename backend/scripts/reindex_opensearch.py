#!/usr/bin/env python3
"""
Backfill OpenSearch from existing `chunks` rows (after enabling OPENSEARCH_URL).

Usage (from `backend/` with venv):

    export DATABASE_URL=...
    export OPENSEARCH_URL=http://localhost:9200
    python scripts/reindex_opensearch.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow `python scripts/reindex_opensearch.py` from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    from sqlalchemy import select

    from app.config import clear_settings_cache, get_settings
    from app.database import get_session_factory
    from app.models import Chunk, Document
    from app.services import search_index

    clear_settings_cache()
    if not (get_settings().opensearch_url or "").strip():
        print("Set OPENSEARCH_URL to reindex.", file=sys.stderr)
        sys.exit(1)

    await search_index.ensure_index_exists()
    factory = get_session_factory()
    async with factory() as session:
        stmt = select(Chunk, Document).join(Document, Chunk.document_id == Document.id)
        rows = (await session.execute(stmt)).all()
    n = 0
    for ch, doc in rows:
        ok = await search_index.index_chunk(
            chunk_id=ch.id,
            document_id=doc.id,
            document_title=doc.title,
            content=ch.content,
        )
        if ok:
            n += 1
    print(f"Indexed {n}/{len(rows)} chunks.")


if __name__ == "__main__":
    asyncio.run(main())
