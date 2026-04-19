"""
Embeddings for RAG.

OpenAI limits:
- **Per request**: total tokens across all inputs (often 300,000).
- **Per input**: max **8,192** tokens per string for embedding models.

Char-based estimates (len/4) can be far below real token counts; we use **tiktoken**
when available so batches stay under limits.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from app.config import get_settings
from app.embedding_dims import MEMORY_ITEM_EMBED_DIM
from app.monitoring.openai_usage import record_embedding_usage
from app.services.langfuse_tracing import openai_trace_kwargs
from app.services.openai_factory import get_async_openai_client

logger = logging.getLogger(__name__)

# Stay safely under API ``max_tokens_per_request`` (~300k).
_MAX_TOKENS_PER_REQUEST = 180_000
# text-embedding-3-* allow 8192 tokens per input; leave margin.
_MAX_INPUT_TOKENS = 8_000

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    def _split_oversized_inputs(text: str) -> list[str]:
        """Split one logical chunk into <= _MAX_INPUT_TOKENS token pieces."""
        ids = _ENC.encode(text)
        if len(ids) <= _MAX_INPUT_TOKENS:
            return [text]
        out: list[str] = []
        for i in range(0, len(ids), _MAX_INPUT_TOKENS):
            out.append(_ENC.decode(ids[i : i + _MAX_INPUT_TOKENS]))
        return out

except ImportError:  # pragma: no cover
    _ENC = None

    def _count_tokens(text: str) -> int:
        # Conservative: real counts often exceed len/4 (underestimate caused 542k batch errors).
        return max(1, len(text) // 2)

    def _split_oversized_inputs(text: str) -> list[str]:
        max_chars = _MAX_INPUT_TOKENS * 2
        if len(text) <= max_chars:
            return [text]
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _mean_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("empty vectors")
    if len(vectors) == 1:
        return vectors[0]
    n = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(n)]


def _expand_for_api(texts: list[str]) -> tuple[list[int], list[str]]:
    """
    Map each logical chunk to one or more API inputs (split if over token limit).
    Returns (orig_indices, flat_pieces) with same length.
    """
    orig_indices: list[int] = []
    flat: list[str] = []
    for i, t in enumerate(texts):
        for piece in _split_oversized_inputs(t):
            orig_indices.append(i)
            flat.append(piece)
    return orig_indices, flat


def _batch_pieces(pieces: list[str]) -> list[list[str]]:
    """Group pieces into API calls where sum(tokens) <= budget."""
    batches: list[list[str]] = []
    current: list[str] = []
    running = 0
    for t in pieces:
        need = _count_tokens(t)
        if need > _MAX_INPUT_TOKENS:
            logger.error("embedding piece exceeds max input tokens after split (%s)", need)
            raise ValueError(f"embedding input too long: {need} tokens")
        if current and running + need > _MAX_TOKENS_PER_REQUEST:
            batches.append(current)
            current = []
            running = 0
        current.append(t)
        running += need
    if current:
        batches.append(current)
    return batches


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    settings = get_settings()
    client = get_async_openai_client()
    model = settings.openai_embedding_model

    orig_indices, pieces = _expand_for_api(texts)
    if not pieces:
        return []

    batches = _batch_pieces(pieces)
    piece_embeddings: list[list[float]] = []
    offset = 0
    for batch in batches:
        resp = await client.embeddings.create(
            model=model,
            input=batch,
            **openai_trace_kwargs(name="embed-texts"),
        )
        await record_embedding_usage(resp, route="embed-texts")
        data = sorted(resp.data, key=lambda d: d.index)
        piece_embeddings.extend(d.embedding for d in data)
        offset += len(batch)

    if len(piece_embeddings) != len(pieces):
        raise RuntimeError(
            f"embedding piece mismatch: got {len(piece_embeddings)}, expected {len(pieces)}"
        )

    # Group by original chunk index and average when a chunk was split
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for orig_i, emb in zip(orig_indices, piece_embeddings, strict=True):
        grouped[orig_i].append(emb)

    out: list[list[float]] = []
    for i in range(len(texts)):
        out.append(_mean_embedding(grouped[i]))

    return out


_ST_MEMORY_MODEL = None


def _get_sentence_transformer_memory():
    """Lazy-load Sentence-Transformers (only when MEMORY_EMBEDDING_PROVIDER=sentence_transformers)."""
    global _ST_MEMORY_MODEL
    if _ST_MEMORY_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "MEMORY_EMBEDDING_PROVIDER=sentence_transformers requires the 'sentence-transformers' "
                "package (pip install sentence-transformers)."
            ) from e
        settings = get_settings()
        _ST_MEMORY_MODEL = SentenceTransformer(settings.sentence_transformer_memory_model)
    return _ST_MEMORY_MODEL


def _encode_memory_sentence_transformers_sync(texts: list[str]) -> list[list[float]]:
    model = _get_sentence_transformer_memory()
    arr = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if getattr(arr, "ndim", 0) == 1:
        out = [arr.tolist()]
    else:
        out = [row.tolist() for row in arr]
    for v in out:
        if len(v) != MEMORY_ITEM_EMBED_DIM:
            raise RuntimeError(
                f"Sentence-Transformers output dimension {len(v)} != {MEMORY_ITEM_EMBED_DIM}. "
                f"Use a {MEMORY_ITEM_EMBED_DIM}-dim model (e.g. all-MiniLM-L6-v2) or update "
                "app.embedding_dims.MEMORY_ITEM_EMBED_DIM and migrate the DB."
            )
    return out


async def _embed_memory_openai_reduced(texts: list[str]) -> list[list[float]]:
    """384-dim memory vectors via OpenAI text-embedding-3-* ``dimensions`` (paid API, not Transformers)."""
    settings = get_settings()
    model = settings.openai_embedding_model
    if not model.startswith("text-embedding-3"):
        raise ValueError(
            f"Memory uses {MEMORY_ITEM_EMBED_DIM}-dim vectors. Set OPENAI_EMBEDDING_MODEL to "
            "text-embedding-3-small, or set MEMORY_EMBEDDING_PROVIDER=sentence_transformers for "
            "free local embeddings."
        )
    client = get_async_openai_client()
    orig_indices, pieces = _expand_for_api(texts)
    if not pieces:
        return []
    batches = _batch_pieces(pieces)
    piece_embeddings: list[list[float]] = []
    for batch in batches:
        resp = await client.embeddings.create(
            model=model,
            input=batch,
            dimensions=MEMORY_ITEM_EMBED_DIM,
            **openai_trace_kwargs(name="embed-memory"),
        )
        await record_embedding_usage(resp, route="embed-memory")
        data = sorted(resp.data, key=lambda d: d.index)
        piece_embeddings.extend(d.embedding for d in data)

    if len(piece_embeddings) != len(pieces):
        raise RuntimeError(
            f"memory embedding piece mismatch: got {len(piece_embeddings)}, expected {len(pieces)}"
        )
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for orig_i, emb in zip(orig_indices, piece_embeddings, strict=True):
        grouped[orig_i].append(emb)
    return [_mean_embedding(grouped[i]) for i in range(len(texts))]


async def embed_memory_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeddings for ``memory_items`` only (384-dim). Document chunks still use :func:`embed_texts`.

    - ``MEMORY_EMBEDDING_PROVIDER=openai``: OpenAI ``text-embedding-3-*`` with ``dimensions=384`` (API cost).
    - ``MEMORY_EMBEDDING_PROVIDER=sentence_transformers``: free local Hugging Face / Sentence-Transformers.

    If Sentence-Transformers fails (e.g. PyTorch version mismatch, ``nn`` errors on load),
    we fall back to OpenAI reduced embeddings so memory retrieval still works.
    """
    if not texts:
        return []
    settings = get_settings()
    if settings.memory_embedding_provider == "sentence_transformers":
        try:
            return await asyncio.to_thread(_encode_memory_sentence_transformers_sync, texts)
        except Exception as e:
            logger.warning(
                "sentence-transformers memory embedding failed (%s); falling back to OpenAI "
                "text-embedding-3 with dimensions=%s",
                e,
                MEMORY_ITEM_EMBED_DIM,
            )
            return await _embed_memory_openai_reduced(texts)
    return await _embed_memory_openai_reduced(texts)
