"""
OpenSearch **native hybrid + RRF**, matching jamwithai/production-agentic-rag-course
``OpenSearchClient._search_hybrid_native`` / ``search_unified``.

Requires an index whose mapping includes a ``knn_vector`` field ``embedding`` (same dim as
``EMBEDDING_DIMENSIONS``). Use a **new** ``OPENSEARCH_INDEX_NAME`` when enabling this; then re-ingest
all documents so each chunk document includes ``embedding``.

Reference: ``reference/.../src/services/opensearch/client.py`` (``search_unified``, ``_search_hybrid_native``),
``index_config_hybrid.py`` (``HYBRID_RRF_PIPELINE``).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

PIPELINE_ID = "hybrid-rrf-pipeline"


def _base_url() -> str | None:
    s = get_settings()
    url = (s.opensearch_url or "").strip().rstrip("/")
    return url or None


def _index_name() -> str:
    return (get_settings().opensearch_index_name or "law_chunks").strip() or "law_chunks"


def _embedding_dim() -> int:
    return int(get_settings().embedding_dimensions)


async def ensure_rrf_pipeline() -> bool:
    """Register the RRF search pipeline (idempotent)."""
    base = _base_url()
    if not base:
        return False
    body: dict[str, Any] = {
        "description": "Post processor for hybrid RRF search (Jam course parity)",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": int(get_settings().rrf_k),
                    }
                }
            }
        ],
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.put(
            f"{base}/_search/pipeline/{PIPELINE_ID}",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code not in (200, 201):
            logger.error("OpenSearch pipeline PUT failed: %s %s", r.status_code, r.text)
            return False
        logger.info("OpenSearch RRF pipeline ready: %s", PIPELINE_ID)
        return True


async def ensure_unified_index_exists() -> bool:
    """
    Create index with ``knn_vector`` embedding field (Jam-style hybrid index).
    Only runs when ``OPENSEARCH_UNIFIED_HYBRID`` is true.
    """
    if not get_settings().opensearch_unified_hybrid:
        return False
    base = _base_url()
    if not base:
        return False
    idx = _index_name()
    dim = _embedding_dim()
    async with httpx.AsyncClient(timeout=30.0) as client:
        head = await client.head(f"{base}/{idx}")
        if head.status_code == 200:
            logger.info("OpenSearch unified index already exists: %s", idx)
            return True

        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "document_title": {"type": "text"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 256, "m": 16},
                        },
                    },
                }
            },
        }
        r = await client.put(f"{base}/{idx}", json=mapping)
        if r.status_code not in (200, 201):
            logger.error("OpenSearch unified index create failed: %s %s", r.status_code, r.text)
            return False
        logger.info("Created OpenSearch unified index: %s (dim=%s)", idx, dim)
        return True


def _bm25_subquery(query: str, scope_document_ids: list[UUID] | None) -> dict[str, Any]:
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
        return {"bool": {"must": [mm], "filter": filt}}
    return mm


async def search_unified_native(
    query: str,
    query_embedding: list[float],
    size: int,
    *,
    scope_document_ids: list[UUID] | None = None,
) -> list[tuple[UUID, float]]:
    """
    Single OpenSearch hybrid query + RRF pipeline → chunk ids and scores.
    Mirrors course ``search_unified`` when embedding is present.
    """
    base = _base_url()
    if not base or not query.strip():
        return []

    bm25_q = _bm25_subquery(query, scope_document_ids)
    k = max(size * 2, 10)
    knn_q = {"knn": {"embedding": {"vector": query_embedding, "k": k}}}
    hybrid_query: dict[str, Any] = {"hybrid": {"queries": [bm25_q, knn_q]}}

    body: dict[str, Any] = {
        "size": size,
        "query": hybrid_query,
        "_source": False,
    }

    idx = _index_name()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{base}/{idx}/_search",
            params={"search_pipeline": PIPELINE_ID},
            json=body,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code != 200:
            logger.warning("OpenSearch unified search failed: %s %s", r.status_code, r.text)
            return []
        data = r.json()
        out: list[tuple[UUID, float]] = []
        for hit in data.get("hits", {}).get("hits", []):
            cid = hit.get("_id")
            score = float(hit.get("_score") or 0.0)
            if cid:
                try:
                    out.append((UUID(str(cid)), score))
                except ValueError:
                    continue
        return out


async def setup_unified_hybrid_resources() -> None:
    """Pipeline + index for Jam-style search (call from app lifespan)."""
    if not get_settings().opensearch_unified_hybrid:
        return
    if not (get_settings().opensearch_url or "").strip():
        return
    await ensure_rrf_pipeline()
    await ensure_unified_index_exists()
