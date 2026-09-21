from sqlalchemy.orm import Session

from m45.personal_context import load_recent_personal_context
from m45.source_history import MessageRole, capture_source_message


def _capture(
    session: Session,
    *,
    conversation_id: str,
    message_id: str,
    content: str,
    source: str = "agent_chat_ui",
    role: MessageRole = "user",
) -> None:
    capture_source_message(
        session,
        source=source,
        conversation_id=conversation_id,
        message_id=message_id,
        role=role,
        content=content,
    )


def test_recent_context_excludes_current_conversation_and_ineligible_sources(
    session: Session,
) -> None:
    _capture(
        session,
        conversation_id="other-1",
        message_id="other-old",
        content="Older context",
    )
    _capture(
        session,
        conversation_id="current",
        message_id="current-message",
        content="Already present in thread state",
    )
    _capture(
        session,
        source="whatsapp_export",
        conversation_id="historical",
        message_id="historical-message",
        content="Stored but not eligible",
    )
    _capture(
        session,
        conversation_id="other-2",
        message_id="other-recent",
        content="Recent context",
        role="assistant",
    )

    messages = load_recent_personal_context(
        session,
        eligible_sources=("agent_chat_ui",),
        current_source="agent_chat_ui",
        current_conversation_id="current",
        message_limit=20,
        character_limit=12_000,
    )

    assert [
        (message.role, message.conversation_id, message.message_id) for message in messages
    ] == [
        ("user", "other-1", "other-old"),
        ("assistant", "other-2", "other-recent"),
    ]


def test_recent_context_keeps_only_the_newest_message_limit(
    session: Session,
) -> None:
    _capture(
        session,
        conversation_id="other-1",
        message_id="message-1",
        content="First",
    )
    _capture(
        session,
        conversation_id="other-2",
        message_id="message-2",
        content="Second",
    )
    _capture(
        session,
        conversation_id="other-3",
        message_id="message-3",
        content="Third",
    )

    messages = load_recent_personal_context(
        session,
        eligible_sources=("agent_chat_ui",),
        current_source="agent_chat_ui",
        current_conversation_id="current",
        message_limit=2,
        character_limit=12_000,
    )

    assert [message.message_id for message in messages] == [
        "message-2",
        "message-3",
    ]


def test_recent_context_stops_before_a_message_exceeding_character_limit(
    session: Session,
) -> None:
    _capture(
        session,
        conversation_id="other-1",
        message_id="older",
        content="1234",
    )
    _capture(
        session,
        conversation_id="other-2",
        message_id="would-exceed-limit",
        content="123456",
    )
    _capture(
        session,
        conversation_id="other-3",
        message_id="newest",
        content="12345",
    )

    messages = load_recent_personal_context(
        session,
        eligible_sources=("agent_chat_ui",),
        current_source="agent_chat_ui",
        current_conversation_id="current",
        message_limit=20,
        character_limit=10,
    )

    assert [message.message_id for message in messages] == ["newest"]
