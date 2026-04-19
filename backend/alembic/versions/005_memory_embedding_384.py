"""Shrink memory_items embeddings to 384-dim (OpenAI reduced or Sentence-Transformers).

Revision ID: 005_mem384
Revises: 004_users
Create Date: 2026-04-18

Existing rows are cleared because stored vectors are not convertible across widths.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005_mem384"
down_revision: Union[str, None] = "004_users"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_IX_HNSW = "ix_memory_items_embedding_hnsw"


def upgrade() -> None:
    op.execute(sa.text(f"DROP INDEX IF EXISTS {_IX_HNSW}"))
    op.execute(sa.text("TRUNCATE TABLE memory_items"))
    op.execute(sa.text("ALTER TABLE memory_items ALTER COLUMN embedding TYPE vector(384)"))
    op.execute(
        sa.text(
            f"""
            CREATE INDEX {_IX_HNSW} ON memory_items
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f"DROP INDEX IF EXISTS {_IX_HNSW}"))
    op.execute(sa.text("TRUNCATE TABLE memory_items"))
    op.execute(sa.text("ALTER TABLE memory_items ALTER COLUMN embedding TYPE vector(1536)"))
    op.execute(
        sa.text(
            f"""
            CREATE INDEX {_IX_HNSW} ON memory_items
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
    )
