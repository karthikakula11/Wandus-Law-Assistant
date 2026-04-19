"""
OpenSearch index for lexical (BM25) and optional **unified hybrid** (Jam course parity).

When ``OPENSEARCH_UNIFIED_HYBRID`` is true, index creation + RRF pipeline are handled in
``opensearch_unified.py`` (knn_vector ``embedding`` field). See course
``src/services/opensearch/client.py`` / ``index_config_hybrid.py``.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


def _index_name() -> str:
    return (get_settings().opensearch_index_name or "law_chunks").strip() or "law_chunks"


def _base_url() -> str | None:
    s = get_settings()
    url = (s.opensearch_url or "").strip().rstrip("/")
    return url or None


async def ensure_index_exists() -> bool:
    """Create index if missing. Delegates to unified hybrid setup when enabled."""
    base = _base_url()
    if not base:
        return False

    settings = get_settings()
    if settings.opensearch_unified_hybrid:
        from app.services.opensearch_unified import ensure_rrf_pipeline, ensure_unified_index_exists

        await ensure_rrf_pipeline()
        return await ensure_unified_index_exists()

    idx = _index_name()
    async with httpx.AsyncClient(timeout=30.0) as client:
        head = await client.head(f"{base}/{idx}")
        if head.status_code == 200:
            return True
        mapping = {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "document_title": {"type": "text"},
                    "content": {"type": "text"},
                }
            },
        }
        r = await client.put(f"{base}/{idx}", json=mapping)
        if r.status_code not in (200, 201):
            logger.error("OpenSearch create index failed: %s %s", r.status_code, r.text)
            return False
        return True


async def index_chunk(
    *,
    chunk_id: UUID,
    document_id: UUID,
    document_title: str,
    content: str,
    embedding: list[float] | None = None,
) -> bool:
    base = _base_url()
    if not base:
        return False
    doc: dict[str, Any] = {
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "document_title": document_title,
        "content": content,
    }
    if embedding is not None and get_settings().opensearch_unified_hybrid:
        doc["embedding"] = embedding
    idx = _index_name()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.put(
            f"{base}/{idx}/_doc/{chunk_id}",
            json=doc,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code not in (200, 201):
            logger.warning("OpenSearch index chunk failed: %s %s", r.status_code, r.text)
            return False
        return True


async def delete_chunk_document(chunk_id: UUID) -> None:
    base = _base_url()
    if not base:
        return
    idx = _index_name()
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.delete(f"{base}/{idx}/_doc/{chunk_id}")


async def search_bm25(
    query: str,
    limit: int,
    *,
    scope_document_ids: list[UUID] | None = None,
) -> list[tuple[UUID, float]]:
    """
    Returns (chunk_id, bm25_score) ordered by score descending.
    OpenSearch returns BM25-style _score.
    """
    base = _base_url()
    if not base or not query.strip():
        return []
    mm: dict[str, Any] = {
        "multi_match": {
            "query": query,
            "fields": ["content^2", "document_title"],
            "type": "best_fields",
        }
    }
    if scope_document_ids:
        if len(scope_document_ids) == 1:
            filt = [{"term": {"document_id": str(scope_document_ids[0])}}]
        else:
            filt = [{"terms": {"document_id": [str(u) for u in scope_document_ids]}}]
        q: dict[str, Any] = {"bool": {"must": [mm], "filter": filt}}
    else:
        q = mm
    body: dict[str, Any] = {
        "size": limit,
        "query": q,
        "_source": False,
    }
    idx = _index_name()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{base}/{idx}/_search",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code != 200:
            logger.warning("OpenSearch search failed: %s %s", r.status_code, r.text)
            return []
        data = r.json()
        out: list[tuple[UUID, float]] = []
        for hit in data.get("hits", {}).get("hits", []):
            cid = hit.get("_id") or (hit.get("_source") or {}).get("chunk_id")
            score = float(hit.get("_score") or 0.0)
            if cid:
                try:
                    out.append((UUID(str(cid)), score))
                except ValueError:
                    continue
        return out


async def ping() -> bool:
    base = _base_url()
    if not base:
        return False
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{base}/")
        return r.status_code == 200
