"""create source messages

Revision ID: 8f8bb4560b93
Revises:
Create Date: 2026-09-19 17:38:06.238950

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f8bb4560b93"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "source_messages",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "captured_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("content <> ''", name="ck_source_messages_content_not_empty"),
        sa.CheckConstraint(
            "conversation_id <> ''", name="ck_source_messages_conversation_id_not_empty"
        ),
        sa.CheckConstraint("message_id <> ''", name="ck_source_messages_message_id_not_empty"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_source_messages_role"),
        sa.CheckConstraint("source <> ''", name="ck_source_messages_source_not_empty"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source", "conversation_id", "message_id", name="uq_source_messages_identity"
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("source_messages")
