from datetime import datetime
from typing import Literal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Identity,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Mapped, Session, mapped_column

from m45.database import Base

type MessageRole = Literal["user", "assistant"]


class SourceMessageIdentityConflict(ValueError):
    """Raised when a source identity is reused with different durable data."""


class SourceMessage(Base):
    __tablename__ = "source_messages"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "conversation_id",
            "message_id",
            name="uq_source_messages_identity",
        ),
        CheckConstraint(
            "source <> ''",
            name="ck_source_messages_source_not_empty",
        ),
        CheckConstraint(
            "conversation_id <> ''",
            name="ck_source_messages_conversation_id_not_empty",
        ),
        CheckConstraint(
            "message_id <> ''",
            name="ck_source_messages_message_id_not_empty",
        ),
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_source_messages_role",
        ),
        CheckConstraint(
            "content <> ''",
            name="ck_source_messages_content_not_empty",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(),
        primary_key=True,
    )
    source: Mapped[str] = mapped_column(Text)
    conversation_id: Mapped[str] = mapped_column(Text)
    message_id: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


def capture_source_message(
    session: Session,
    *,
    source: str,
    conversation_id: str,
    message_id: str,
    role: MessageRole,
    content: str,
) -> SourceMessage:
    statement = (
        insert(SourceMessage)
        .values(
            source=source,
            conversation_id=conversation_id,
            message_id=message_id,
            role=role,
            content=content,
        )
        .on_conflict_do_nothing(
            constraint="uq_source_messages_identity",
        )
        .returning(SourceMessage.id)
    )

    inserted_id = session.scalar(statement)

    if inserted_id is not None:
        return session.get_one(SourceMessage, inserted_id)

    existing = session.scalars(
        select(SourceMessage).where(
            SourceMessage.source == source,
            SourceMessage.conversation_id == conversation_id,
            SourceMessage.message_id == message_id,
        )
    ).one()

    if existing.role != role or existing.content != content:
        raise SourceMessageIdentityConflict(
            "source message identity already exists with different role or content: "
            f"source={source!r}, "
            f"conversation_id={conversation_id!r}, "
            f"message_id={message_id!r}"
        )

    return existing
