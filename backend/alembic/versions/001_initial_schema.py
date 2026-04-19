"""Initial documents and chunks with pgvector

Revision ID: 001_initial
Revises:
Create Date: 2026-04-15

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBED_DIM = 1536


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("source_uri", sa.String(2048), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
