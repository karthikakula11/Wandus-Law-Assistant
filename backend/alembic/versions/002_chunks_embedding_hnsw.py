"""HNSW index on chunks.embedding for approximate cosine nearest-neighbor search.

Revision ID: 002_hnsw
Revises: 001_initial
Create Date: 2026-04-15

Uses pgvector HNSW with vector_cosine_ops to match Chunk.embedding.cosine_distance()
in retrieval (rag.py, retrieval_hybrid.py).

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_hnsw"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Tunable: higher m / ef_construction → better recall, more index size & build time.
# Defaults are a reasonable balance for law-corpus scale.
_HNSW_INDEX = "ix_chunks_embedding_hnsw"


def upgrade() -> None:
    op.execute(
        sa.text(
            f"""
            CREATE INDEX {_HNSW_INDEX} ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
    )


def downgrade() -> None:
    op.drop_index(_HNSW_INDEX, table_name="chunks")
