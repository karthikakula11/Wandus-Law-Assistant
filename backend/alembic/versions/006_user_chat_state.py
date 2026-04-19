"""Persist sidebar chat threads per memory user (JSON snapshot).

Revision ID: 006_chat_state
Revises: 005_mem384
Create Date: 2026-04-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "006_chat_state"
down_revision: Union[str, None] = "005_mem384"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_chat_states",
        sa.Column("memory_user_key", sa.String(64), primary_key=True, nullable=False),
        sa.Column("payload", JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("user_chat_states")
