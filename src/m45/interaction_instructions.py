from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Integer,
    Text,
    UniqueConstraint,
    func,
    select,
    text,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from m45.database import Base

DEFAULT_INTERACTION_INSTRUCTIONS = (
    "Pay attention to the user's life, relationships, plans, preferences, and "
    "ongoing situations. Bring up prior context when it is naturally useful. Ask "
    "a focused follow-up only when it would meaningfully improve the conversation "
    "or resolve important uncertainty; do not force a question into every response. "
    "Answer direct questions directly.\n\n"
    "Be concise by default and leave room for the user to respond."
)


class InteractionInstructionRevision(Base):
    __tablename__ = "interaction_instruction_revisions"
    __table_args__ = (
        CheckConstraint(
            "instructions <> ''",
            name="ck_interaction_instruction_revisions_nonempty",
        ),
        UniqueConstraint(
            "tool_call_id",
            name="uq_interaction_instruction_revisions_tool_call_id",
        ),
    )

    revision: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=False,
    )
    instructions: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    conversation_id: Mapped[str] = mapped_column(Text)
    message_id: Mapped[str] = mapped_column(Text)
    tool_call_id: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


def load_current_interaction_instructions(session: Session) -> tuple[int, str]:
    latest = session.scalar(
        select(InteractionInstructionRevision)
        .order_by(InteractionInstructionRevision.revision.desc())
        .limit(1)
    )
    if latest is None:
        return 0, DEFAULT_INTERACTION_INSTRUCTIONS

    return latest.revision, latest.instructions


INTERACTION_INSTRUCTIONS_LOCK_KEY = 0x6D3435  # "m45"


def save_interaction_instruction_revision(
    session: Session,
    *,
    expected_revision: int,
    instructions: str,
    source: str,
    conversation_id: str,
    message_id: str,
    tool_call_id: str,
) -> int:
    if expected_revision < 0:
        raise ValueError("expected revision must not be negative")
    if not instructions.strip():
        raise ValueError("interaction instructions must not be blank")

    session.execute(
        text("SELECT pg_advisory_xact_lock(:key)"),
        {"key": INTERACTION_INSTRUCTIONS_LOCK_KEY},
    )

    existing = session.scalar(
        select(InteractionInstructionRevision).where(
            InteractionInstructionRevision.tool_call_id == tool_call_id
        )
    )

    if existing is not None:
        if (
            existing.revision != expected_revision + 1
            or existing.instructions != instructions
            or existing.source != source
            or existing.conversation_id != conversation_id
            or existing.message_id != message_id
        ):
            raise ValueError("tool call ID was reused for a different revision")
        return existing.revision

    current_revision, _ = load_current_interaction_instructions(session)
    if current_revision != expected_revision:
        raise ValueError(
            f"interaction instructions changed: expected revision "
            f"{expected_revision}, current revision {current_revision}"
        )

    revision = current_revision + 1
    session.add(
        InteractionInstructionRevision(
            revision=revision,
            instructions=instructions,
            source=source,
            conversation_id=conversation_id,
            message_id=message_id,
            tool_call_id=tool_call_id,
        )
    )
    session.flush()
    return revision
