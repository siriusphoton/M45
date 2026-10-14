import asyncio
from datetime import date, datetime

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from sqlalchemy.orm import Session

from m45.daily_log import draft_daily_log, load_daily_source_messages
from m45.source_history import SourceMessage, capture_source_message


def test_daily_source_messages_respect_window_sources_and_cutoff(
    session: Session,
) -> None:
    cases = [
        ("next-start", "discord", "2026-10-09T23:30:00+00:00"),
        ("late", "agent_chat_ui", "2026-10-09T23:29:59+00:00"),
        ("noon", "discord", "2026-10-09T06:30:00+00:00"),
        ("start", "discord", "2026-10-08T23:30:00+00:00"),
        ("before", "discord", "2026-10-08T23:29:59+00:00"),
        ("excluded", "whatsapp_export", "2026-10-09T10:00:00+00:00"),
    ]

    for message_id, source, timestamp in cases:
        message = capture_source_message(
            session,
            source=source,
            conversation_id=f"conversation-{message_id}",
            message_id=message_id,
            role="assistant" if message_id == "late" else "user",
            content=f"Test message: {message_id}",
        )
        message.captured_at = datetime.fromisoformat(timestamp)

    session.flush()

    messages = load_daily_source_messages(
        session,
        day=date(2026, 10, 9),
        eligible_sources=("discord", "agent_chat_ui"),
    )

    assert [(message.message_id, message.role) for message in messages] == [
        ("start", "user"),
        ("noon", "user"),
        ("late", "assistant"),
    ]

    morning_messages = load_daily_source_messages(
        session,
        day=date(2026, 10, 9),
        eligible_sources=("discord", "agent_chat_ui"),
        before=datetime.fromisoformat("2026-10-09T06:30:00+00:00"),
    )

    assert [message.message_id for message in morning_messages] == ["start"]


def test_daily_writer_rejects_empty_model_output() -> None:
    message = SourceMessage(
        id=1,
        source="discord",
        conversation_id="test-conversation",
        message_id="test-message",
        role="user",
        content="I finished the report.",
        captured_at=datetime.fromisoformat("2026-10-09T12:00:00+05:30"),
    )
    model = GenericFakeChatModel(messages=iter([" \n "]))

    with pytest.raises(ValueError, match="daily writer returned no text"):
        asyncio.run(
            draft_daily_log(
                model,
                day=date(2026, 10, 9),
                as_of=datetime.fromisoformat("2026-10-09T13:00:00+05:30"),
                messages=[message],
            )
        )
