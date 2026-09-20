import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from m45.source_history import (
    SourceMessage,
    SourceMessageIdentityConflict,
    capture_source_message,
)


def test_capture_source_message_is_idempotent(session: Session) -> None:
    first = capture_source_message(
        session,
        source="agent_chat_ui",
        conversation_id="thread-1",
        message_id="message-1",
        role="user",
        content="Hello",
    )
    repeated = capture_source_message(
        session,
        source="agent_chat_ui",
        conversation_id="thread-1",
        message_id="message-1",
        role="user",
        content="Hello",
    )

    assert repeated.id == first.id
    assert session.scalar(select(func.count()).select_from(SourceMessage)) == 1


def test_capture_source_message_rejects_changed_data_for_existing_identity(
    session: Session,
) -> None:
    capture_source_message(
        session,
        source="agent_chat_ui",
        conversation_id="thread-1",
        message_id="message-1",
        role="user",
        content="Original",
    )

    with pytest.raises(SourceMessageIdentityConflict):
        capture_source_message(
            session,
            source="agent_chat_ui",
            conversation_id="thread-1",
            message_id="message-1",
            role="user",
            content="Changed",
        )

    assert session.scalar(select(func.count()).select_from(SourceMessage)) == 1


def test_capture_source_message_preserves_distinct_message_ids(
    session: Session,
) -> None:
    first = capture_source_message(
        session,
        source="agent_chat_ui",
        conversation_id="thread-1",
        message_id="message-1",
        role="user",
        content="Same text",
    )
    second = capture_source_message(
        session,
        source="agent_chat_ui",
        conversation_id="thread-1",
        message_id="message-2",
        role="user",
        content="Same text",
    )

    assert second.id != first.id
    assert session.scalar(select(func.count()).select_from(SourceMessage)) == 2
