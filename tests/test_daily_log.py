import asyncio
from datetime import date, datetime

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from m45.daily_log import (
    DailyLogRevision,
    DailyLogRevisionSource,
    append_daily_log_revision,
    draft_daily_log,
    load_daily_source_messages,
    load_latest_daily_log_revision,
)
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
        processing_day=date(2026, 10, 9),
        eligible_sources=("discord", "agent_chat_ui"),
    )

    assert [(message.message_id, message.role) for message in messages] == [
        ("start", "user"),
        ("noon", "user"),
        ("late", "assistant"),
    ]

    morning_messages = load_daily_source_messages(
        session,
        processing_day=date(2026, 10, 9),
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
                processing_day=date(2026, 10, 9),
                as_of=datetime.fromisoformat("2026-10-09T13:00:00+05:30"),
                messages=[message],
            )
        )


def test_daily_log_revisions_preserve_history_and_exact_source_membership(
    session: Session,
) -> None:
    processing_day = date(2026, 10, 1)
    cutoff = datetime.fromisoformat("2026-10-01T14:00:00+05:30")
    original_message = capture_source_message(
        session,
        source="discord",
        conversation_id="daily-log-test",
        message_id="daily-0",
        role="user",
        content="The call is at 16:00.",
    )
    original_message.captured_at = datetime.fromisoformat("2026-10-01T12:00:00+05:30")
    session.flush()

    assert load_latest_daily_log_revision(session, processing_day=processing_day) is None
    first_source_id = original_message.id
    first = append_daily_log_revision(
        session,
        processing_day=processing_day,
        base_revision_number=0,
        content="Call planned for 16:00.",
        as_of=cutoff,
        source_message_ids={first_source_id},
    )

    # This source arrives after revision 1 but predates its review cutoff.
    late_message = capture_source_message(
        session,
        source="discord",
        conversation_id="daily-log-test",
        message_id="daily-1",
        role="user",
        content="It moved to 17:00.",
    )
    late_message.captured_at = datetime.fromisoformat("2026-10-01T13:00:00+05:30")
    session.flush()

    messages = load_daily_source_messages(
        session,
        processing_day=processing_day,
        eligible_sources=("discord", "agent_chat_ui"),
        before=cutoff,
    )
    source_ids = {message.id for message in messages}
    assert source_ids == {first_source_id, late_message.id}

    second = append_daily_log_revision(
        session,
        processing_day=processing_day,
        base_revision_number=1,
        content="Call moved from 16:00 to 17:00.",
        as_of=cutoff,
        source_message_ids=source_ids,
    )
    expected_links = {(first.id, first_source_id)} | {
        (second.id, source_id) for source_id in source_ids
    }

    with pytest.raises(ValueError, match="daily log changed"):
        append_daily_log_revision(
            session,
            processing_day=processing_day,
            base_revision_number=1,
            content="A stale replacement.",
            as_of=cutoff,
            source_message_ids=source_ids,
        )

    with pytest.raises(RuntimeError, match="cancel update"), session.begin_nested():
        append_daily_log_revision(
            session,
            processing_day=processing_day,
            base_revision_number=2,
            content="An update that must be rolled back.",
            as_of=cutoff,
            source_message_ids=source_ids,
        )
        raise RuntimeError("cancel update")

    session.expire_all()
    latest_log_revision = load_latest_daily_log_revision(session, processing_day=processing_day)
    assert latest_log_revision is not None
    assert (latest_log_revision.revision_number, latest_log_revision.content) == (
        2,
        "Call moved from 16:00 to 17:00.",
    )
    assert list(
        session.execute(
            select(DailyLogRevision.revision_number, DailyLogRevision.content)
            .where(DailyLogRevision.processing_day == processing_day)
            .order_by(DailyLogRevision.revision_number)
        )
    ) == [(1, "Call planned for 16:00."), (2, "Call moved from 16:00 to 17:00.")]
    assert (
        set(
            session.execute(
                select(
                    DailyLogRevisionSource.daily_log_revision_id,
                    DailyLogRevisionSource.source_message_id,
                )
            )
        )
        == expected_links
    )
