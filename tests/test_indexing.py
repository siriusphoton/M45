from datetime import date, datetime, timedelta

from m45.daily_log import DailyLogRevision
from m45.indexing import build_daily_log_passages, build_source_passages
from m45.source_history import MessageRole, SourceMessage


def test_daily_log_passages_keep_section_and_revision_references() -> None:
    previous_log_revision = DailyLogRevision(
        id=41,
        processing_day=date(2026, 10, 1),
        revision_number=1,
        content="# Daily log\n\n## Meals\nLunch planned at the office.",
    )
    latest_log_revision = DailyLogRevision(
        id=42,
        processing_day=previous_log_revision.processing_day,
        revision_number=2,
        content=(
            "# Daily log\n\n## Meals\nAte biryani at the office.\n\n"
            "## Tasks and plans\nCompleted the report."
        ),
    )

    old_passages = build_daily_log_passages(previous_log_revision)
    passages = build_daily_log_passages(latest_log_revision)

    assert {item.id for item in old_passages}.isdisjoint(item.id for item in passages)
    assert [item.metadata["section"] for item in passages] == ["Meals", "Tasks and plans"]
    assert all(
        item.metadata["daily_log_revision_id"] == 42
        and item.metadata["revision_number"] == 2
        and item.metadata["processing_day"] == "2026-10-01"
        for item in passages
    )
    assert "Ate biryani" in passages[0].page_content
    assert "Completed the report" not in passages[0].page_content
    assert "Completed the report" in passages[1].page_content
    assert "Lunch planned" not in "\n".join(item.page_content for item in passages)


def test_long_log_sections_keep_context_and_all_words() -> None:
    words = [f"detail{number}" for number in range(150)]
    log_revision = DailyLogRevision(
        id=43,
        processing_day=date(2026, 10, 1),
        revision_number=3,
        content="# Daily log\n\n## Phone research\n" + " ".join(words),
    )

    passages = build_daily_log_passages(log_revision, chunk_size=180, chunk_overlap=30)

    assert len(passages) > 1
    bodies: list[str] = []
    for item in passages:
        header, body = item.page_content.split("\n\n", maxsplit=1)
        assert header == "Daily log for processing day 2026-10-01\nSection: Phone research"
        assert item.metadata["daily_log_revision_id"] == 43
        assert len(body) <= 180
        bodies.append(body)

    assert set(" ".join(bodies).split()) == set(words)


def _message(
    row_id: int,
    content: str,
    *,
    source: str = "discord",
    conversation_id: str = "chat",
    role: MessageRole = "user",
) -> SourceMessage:
    return SourceMessage(
        id=row_id,
        source=source,
        conversation_id=conversation_id,
        message_id=str(row_id),
        role=role,
        content=content,
        captured_at=datetime.fromisoformat("2026-10-01T12:00:00+05:30") + timedelta(seconds=row_id),
    )


def test_source_passages_preserve_dialogue_without_crossing_conversations() -> None:
    opening = "I spent the afternoon finishing the report at work."
    question = "Did you call Harsha?"
    messages = [
        _message(1, opening),
        _message(2, question, role="assistant"),
        _message(3, "yes"),
        _message(4, "yes", conversation_id="other"),
        _message(5, "yes", source="agent_chat_ui"),
    ]

    passages = build_source_passages(list(reversed(messages)), chunk_size=200)
    by_membership = {tuple(item.metadata["source_message_ids"]): item for item in passages}

    assert set(by_membership) == {(1, 2), (2, 3), (4,), (5,)}
    reply = by_membership[(2, 3)]
    assert reply.metadata["source"] == "discord"
    assert reply.metadata["conversation_id"] == "chat"
    assert "assistant: Did you call Harsha?" in reply.page_content
    assert "user: yes" in reply.page_content
    assert "Thursday, 01 October 2026, 12:00:03 IST" in reply.page_content
    assert by_membership[(5,)].metadata["source"] == "agent_chat_ui"


def test_oversized_source_message_keeps_attribution_and_all_words() -> None:
    words = [f"detail{number}" for number in range(100)]
    message = _message(10, " ".join(words), role="assistant")

    passages = build_source_passages([message], chunk_size=100)

    assert len(passages) > 1
    assert len({item.id for item in passages}) == len(passages)
    bodies: list[str] = []
    for item in passages:
        assert item.metadata["source_message_ids"] == [10]
        assert "Thursday, 01 October 2026, 12:00:10 IST" in item.page_content
        _, block = item.page_content.split("\n\n", maxsplit=1)
        assert len(block) <= 100
        _, body = block.split("assistant: ", maxsplit=1)
        bodies.append(body)

    assert set(" ".join(bodies).split()) == set(words)
