"""create interaction instruction revisions

Revision ID: 0f77ca701706
Revises: 8f8bb4560b93
Create Date: 2026-09-30 18:07:26.921747

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0f77ca701706"
down_revision: str | Sequence[str] | None = "8f8bb4560b93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create append-only approved interaction-instruction revisions."""
    op.create_table(
        "interaction_instruction_revisions",
        sa.Column("revision", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("instructions", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("tool_call_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "instructions <> ''",
            name="ck_interaction_instruction_revisions_nonempty",
        ),
        sa.PrimaryKeyConstraint("revision"),
        sa.UniqueConstraint(
            "tool_call_id",
            name="uq_interaction_instruction_revisions_tool_call_id",
        ),
    )


def downgrade() -> None:
    """Remove interaction-instruction revisions."""
    op.drop_table("interaction_instruction_revisions")
