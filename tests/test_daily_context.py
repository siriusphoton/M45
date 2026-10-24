from datetime import date, datetime
from typing import cast

import pytest
from sqlalchemy.orm import Session

from m45.daily_context import format_daily_context, load_daily_context
from m45.daily_log import DailyLogRevision, DailyLogRevisionSource
from m45.source_history import SourceMessage


def test_daily_context_preserves_uncovered_evidence_and_exact_active_identity(
    session: Session,
) -> None:
    cases = [
        ("previous", "discord", "current", "2026-09-30T18:00:00+05:30"),
        ("covered", "discord", "current", "2026-10-01T20:00:00+05:30"),
        ("active", "discord", "current", "2026-10-02T01:00:00+05:30"),
        ("active", "discord", "other", "2026-10-02T01:10:00+05:30"),
        ("active", "agent_chat_ui", "current", "2026-10-02T01:20:00+05:30"),
        ("future", "discord", "current", "2026-10-02T04:00:00+05:30"),
    ]
    messages: list[SourceMessage] = []
    for message_id, source, conversation_id, timestamp in cases:
        message = SourceMessage(
            source=source,
            conversation_id=conversation_id,
            message_id=message_id,
            role="user",
            content=f"{source}/{conversation_id}/{message_id}",
            captured_at=datetime.fromisoformat(timestamp),
        )
        session.add(message)
        messages.append(message)
    first = DailyLogRevision(
        processing_day=date(2026, 10, 1),
        revision_number=1,
        content="Earlier narrative.",
        as_of=datetime.fromisoformat("2026-10-01T21:00:00+05:30"),
    )
    latest = DailyLogRevision(
        processing_day=first.processing_day,
        revision_number=2,
        content="Current narrative.",
        as_of=datetime.fromisoformat("2026-10-01T22:00:00+05:30"),
    )
    session.add_all([first, latest])
    session.flush()
    session.add_all(
        [
            DailyLogRevisionSource(daily_log_revision_id=log.id, source_message_id=messages[1].id)
            for log in (first, latest)
        ]
    )
    # Arrives after the log save but its capture time predates that log's cutoff.
    late = SourceMessage(
        source="discord",
        conversation_id="current",
        message_id="late",
        role="assistant",
        content="A later-arriving correction.",
        captured_at=datetime.fromisoformat("2026-10-01T21:30:00+05:30"),
    )
    session.add(late)
    session.flush()

    context = load_daily_context(
        session,
        now=datetime.fromisoformat("2026-10-02T03:00:00+05:30"),
        eligible_sources=("discord", "agent_chat_ui"),
        current_source="discord",
        current_conversation_id="current",
        active_message_ids=frozenset({"active"}),
    )
    assert context["current_processing_day"] == "2026-10-01"
    days = cast(list[dict[str, object]], context["days"])
    assert [day["processing_day"] for day in days] == ["2026-09-30", "2026-10-01"]
    assert days[0]["daily_log"] is None
    previous_sources = cast(list[dict[str, object]], days[0]["additional_source_messages"])
    assert [row["source_message_id"] for row in previous_sources] == [messages[0].id]
    log = cast(dict[str, object], days[1]["daily_log"])
    assert log["daily_log_revision_id"] == latest.id
    additional = cast(list[dict[str, object]], days[1]["additional_source_messages"])
    assert [row["source_message_id"] for row in additional] == [
        late.id,
        messages[3].id,
        messages[4].id,
    ]
    assert additional[0]["role"] == "assistant"
    assert days[1]["coverage"] == {
        "eligible_source_count": 5,
        "writer_input_count": 1,
        "uncovered_in_active_conversation": 1,
        "additional_source_count": 3,
    }
    assert "A later-arriving correction." in format_daily_context(context)

    rollover = load_daily_context(
        session,
        now=datetime.fromisoformat("2026-10-02T05:00:00+05:30"),
        eligible_sources=("discord", "agent_chat_ui"),
        current_source="discord",
        current_conversation_id="current",
        active_message_ids=frozenset({"active"}),
    )
    days = cast(list[dict[str, object]], rollover["days"])
    assert [day["processing_day"] for day in days] == ["2026-10-01", "2026-10-02"]
    additional = cast(list[dict[str, object]], days[0]["additional_source_messages"])
    assert additional[-1]["source_message_id"] == messages[5].id
    assert days[1]["daily_log"] is None and days[1]["additional_source_messages"] == []

    with pytest.raises(ValueError, match="extends beyond now"):
        load_daily_context(
            session,
            now=datetime.fromisoformat("2026-10-01T21:30:00+05:30"),
            eligible_sources=("discord", "agent_chat_ui"),
            current_source="discord",
            current_conversation_id="current",
            active_message_ids=frozenset(),
        )
