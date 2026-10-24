"""Clarify daily-log column and passage metadata names.

Revision ID: c7e2a91d4f60
Revises: b6db50a79b6f
Create Date: 2026-10-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7e2a91d4f60"
down_revision: str | Sequence[str] | None = "b6db50a79b6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rename fields in place, preserving revision IDs, links, text and vectors."""
    op.alter_column(
        "daily_log_revisions", "day", new_column_name="processing_day", existing_type=sa.Date()
    )
    op.alter_column(
        "daily_log_revisions",
        "revision",
        new_column_name="revision_number",
        existing_type=sa.Integer(),
    )
    op.alter_column(
        "daily_log_revision_sources",
        "revision_id",
        new_column_name="daily_log_revision_id",
        existing_type=sa.BigInteger(),
    )
    op.execute("""
        UPDATE search_passages
        SET details = (details - 'day')
            || jsonb_build_object('processing_day', details -> 'day')
        WHERE daily_log_revision_id IS NOT NULL AND details ? 'day'
    """)
    op.execute("""
        UPDATE search_passages
        SET details = (details - 'revision')
            || jsonb_build_object('revision_number', details -> 'revision')
        WHERE daily_log_revision_id IS NOT NULL AND details ? 'revision'
    """)
    op.execute("""
        UPDATE search_passages
        SET embedding_config = (embedding_config - 'documented_format')
            || jsonb_build_object('document_format', embedding_config -> 'documented_format')
        WHERE embedding_config ? 'documented_format'
    """)


def downgrade() -> None:
    """Restore the previous names without removing revision or search data."""
    op.execute("""
        UPDATE search_passages
        SET embedding_config = (embedding_config - 'document_format')
            || jsonb_build_object('documented_format', embedding_config -> 'document_format')
        WHERE embedding_config ? 'document_format'
    """)
    op.execute("""
        UPDATE search_passages
        SET details = (details - 'revision_number')
            || jsonb_build_object('revision', details -> 'revision_number')
        WHERE daily_log_revision_id IS NOT NULL AND details ? 'revision_number'
    """)
    op.execute("""
        UPDATE search_passages
        SET details = (details - 'processing_day')
            || jsonb_build_object('day', details -> 'processing_day')
        WHERE daily_log_revision_id IS NOT NULL AND details ? 'processing_day'
    """)
    op.alter_column(
        "daily_log_revision_sources",
        "daily_log_revision_id",
        new_column_name="revision_id",
        existing_type=sa.BigInteger(),
    )
    op.alter_column(
        "daily_log_revisions",
        "revision_number",
        new_column_name="revision",
        existing_type=sa.Integer(),
    )
    op.alter_column(
        "daily_log_revisions", "processing_day", new_column_name="day", existing_type=sa.Date()
    )
