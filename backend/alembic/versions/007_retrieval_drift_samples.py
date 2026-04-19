"""Retrieval confidence samples for drift detection (KS vs prior window).

Revision ID: 007_drift_samples
Revises: 006_chat_state
Create Date: 2026-04-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "007_drift_samples"
down_revision: Union[str, None] = "006_chat_state"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "retrieval_drift_samples",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_retrieval_drift_samples_created_at",
        "retrieval_drift_samples",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_retrieval_drift_samples_created_at", table_name="retrieval_drift_samples")
    op.drop_table("retrieval_drift_samples")
