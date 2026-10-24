from datetime import date, datetime, timedelta
from typing import cast

from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45.daily_log import DailyLogRevision
from m45.retrieval_tools import (
    create_daily_log_browse_tool,
    create_source_conversation_browse_tool,
)
from m45.source_history import SourceMessage


def test_browsing_keeps_current_logs_and_pages_within_a_conversation(engine: Engine) -> None:
    start_day = date(2026, 10, 1)
    end_day = start_day + timedelta(days=10)
    timestamp = datetime.fromisoformat("2026-10-01T12:00:00+05:30")
    conversation_id = "browse-test"
    other_conversation_id = "browse-other-test"
    try:
        with Session(engine) as session, session.begin():
            for offset in range(11):
                session.add(
                    DailyLogRevision(
                        processing_day=start_day + timedelta(days=offset),
                        revision_number=1,
                        content="Original account.",
                        as_of=timestamp + timedelta(days=offset),
                    )
                )
            latest = DailyLogRevision(
                processing_day=start_day,
                revision_number=2,
                content="Corrected account. " * 30,
                as_of=timestamp,
            )
            session.add(latest)
            messages: list[SourceMessage] = []
            for index in range(12):
                message = SourceMessage(
                    source="discord",
                    conversation_id=conversation_id,
                    message_id=str(index),
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"Message {index}",
                    captured_at=timestamp,
                )
                messages.append(message)
                session.add(message)
            session.flush()
            ids = [message.id for message in messages]
            latest_id = latest.id
            # Inserted later, but chronologically before all the tied messages.
            older = SourceMessage(
                source="discord",
                conversation_id=conversation_id,
                message_id="older",
                role="user",
                content="Earlier report.",
                captured_at=timestamp - timedelta(minutes=1),
            )
            session.add_all(
                [
                    older,
                    SourceMessage(
                        source="agent_chat_ui",
                        conversation_id=conversation_id,
                        message_id="other-surface",
                        role="user",
                        content="Unrelated surface.",
                        captured_at=timestamp,
                    ),
                    SourceMessage(
                        source="discord",
                        conversation_id=other_conversation_id,
                        message_id="other-chat",
                        role="user",
                        content="Unrelated conversation.",
                        captured_at=timestamp,
                    ),
                ]
            )
            session.flush()
            older_id = older.id

        browse_logs = create_daily_log_browse_tool(engine)
        first = cast(
            dict[str, object],
            browse_logs.invoke(
                {
                    "start_day": start_day.isoformat(),
                    "end_day": end_day.isoformat(),
                }
            ),
        )
        logs = cast(list[dict[str, object]], first["logs"])
        assert len(logs) == 10
        assert logs[0]["daily_log_revision_id"] == latest_id
        assert logs[0]["truncated"] is True
        assert len(str(logs[0]["preview"])) == 300
        second = cast(
            dict[str, object],
            browse_logs.invoke(
                {
                    "start_day": first["next_start_day"],
                    "end_day": end_day.isoformat(),
                }
            ),
        )
        logs += cast(list[dict[str, object]], second["logs"])
        assert [item["processing_day"] for item in logs] == [
            (start_day + timedelta(days=offset)).isoformat() for offset in range(11)
        ]
        assert second["next_start_day"] is None

        browse_sources = create_source_conversation_browse_tool(engine)
        forward = cast(
            dict[str, object],
            browse_sources.invoke(
                {
                    "source_message_id": older_id,
                    "direction": "after",
                }
            ),
        )
        rows = cast(list[dict[str, object]], forward["messages"])
        assert [row["source_message_id"] for row in rows] == ids[:10]
        assert [row["role"] for row in rows[:2]] == ["user", "assistant"]
        tail = cast(
            dict[str, object],
            browse_sources.invoke(
                {
                    "source_message_id": forward["next_anchor_id"],
                    "direction": "after",
                }
            ),
        )
        assert [
            row["source_message_id"] for row in cast(list[dict[str, object]], tail["messages"])
        ] == ids[10:]
        assert tail["next_anchor_id"] is None

        backward = cast(
            dict[str, object],
            browse_sources.invoke(
                {
                    "source_message_id": ids[-1],
                    "direction": "before",
                }
            ),
        )
        assert [
            row["source_message_id"] for row in cast(list[dict[str, object]], backward["messages"])
        ] == ids[1:11]
        head = cast(
            dict[str, object],
            browse_sources.invoke(
                {
                    "source_message_id": backward["next_anchor_id"],
                    "direction": "before",
                }
            ),
        )
        assert [
            row["source_message_id"] for row in cast(list[dict[str, object]], head["messages"])
        ] == [older_id, ids[0]]
        assert head["next_anchor_id"] is None
    finally:
        with Session(engine) as session, session.begin():
            session.execute(
                delete(DailyLogRevision).where(
                    DailyLogRevision.processing_day >= start_day,
                    DailyLogRevision.processing_day <= end_day,
                )
            )
            session.execute(
                delete(SourceMessage).where(
                    SourceMessage.conversation_id.in_([conversation_id, other_conversation_id])
                )
            )
