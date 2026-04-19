"""App users table + seed primary user (for memory key linkage).

Revision ID: 004_users
Revises: 003_memory
Create Date: 2026-04-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "004_users"
down_revision: Union[str, None] = "003_memory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Stable id for docs / clients (deterministic seed).
_SEED_USER_ID = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
_SEED_MEMORY_KEY = "pintu_owner_01"


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(128), nullable=False),
        sa.Column("display_name", sa.String(256), nullable=True),
        sa.Column("memory_user_key", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_memory_user_key", "users", ["memory_user_key"], unique=True)

    # Stable seed so clients can document / default localStorage to this memory key.
    op.execute(
        sa.text(
            f"""
            INSERT INTO users (id, username, display_name, memory_user_key)
            VALUES (
                '{_SEED_USER_ID}'::uuid,
                'owner',
                'Primary user',
                '{_SEED_MEMORY_KEY}'
            )
            """
        )
    )


def downgrade() -> None:
    op.drop_index("ix_users_memory_user_key", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
