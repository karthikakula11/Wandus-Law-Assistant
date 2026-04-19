"""Long-term memory_items (pgvector) for per-user chat facts

Revision ID: 003_memory
Revises: 002_hnsw
Create Date: 2026-04-17

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

revision: str = "003_memory"
down_revision: Union[str, None] = "002_hnsw"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBED_DIM = 1536
_IX_HNSW = "ix_memory_items_embedding_hnsw"


def upgrade() -> None:
    op.create_table(
        "memory_items",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_key", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_memory_items_user_key", "memory_items", ["user_key"])
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
    op.drop_index(_IX_HNSW, table_name="memory_items")
    op.drop_index("ix_memory_items_user_key", table_name="memory_items")
    op.drop_table("memory_items")
