"""Vector sizes for pgvector columns (must match Alembic migrations)."""

# Document chunks — full OpenAI text-embedding-3-small
CHUNK_EMBED_DIM = 1536

# Long-term memory — 384-dim (OpenAI ``dimensions=384`` or Sentence-Transformers MiniLM)
MEMORY_ITEM_EMBED_DIM = 384
