from datetime import date, datetime

import pytest
from langchain_ollama import OllamaEmbeddings
from sqlalchemy import func, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session

from m45.daily_log import DailyLogRevision
from m45.search_index import (
    SearchPassage,
    embed_daily_log_passages,
    read_daily_log_page,
    read_source_message_page,
    replace_daily_log_passages,
    replace_source_passages,
    search_hybrid_passages,
    search_lexical_passages,
    search_semantic_passages,
)
from m45.source_history import capture_source_message


def test_log_passage_replacement_is_atomic_and_preserves_canonical_revisions(
    session: Session,
) -> None:
    first = DailyLogRevision(
        processing_day=date(2026, 10, 1),
        revision_number=1,
        content="## Tasks\nCall planned.\n\n## Meals\nLunch at home.",
        as_of=datetime.fromisoformat("2026-10-01T20:00:00+05:30"),
    )
    other = DailyLogRevision(
        processing_day=date(2026, 10, 2),
        revision_number=1,
        content="## Tasks\nRead a book.",
        as_of=datetime.fromisoformat("2026-10-02T20:00:00+05:30"),
    )
    session.add_all([first, other])
    session.flush()
    assert replace_daily_log_passages(session, processing_day=first.processing_day) == 2
    assert replace_daily_log_passages(session, processing_day=other.processing_day) == 1
    original_ids = set(session.scalars(select(SearchPassage.id)))

    second = DailyLogRevision(
        processing_day=first.processing_day,
        revision_number=2,
        content="## Tasks\nCall completed.",
        as_of=first.as_of,
    )
    session.add(second)
    session.flush()

    with pytest.raises(RuntimeError, match="cancel replacement"), session.begin_nested():
        replace_daily_log_passages(session, processing_day=first.processing_day)
        raise RuntimeError("cancel replacement")

    assert set(session.scalars(select(SearchPassage.id))) == original_ids
    assert replace_daily_log_passages(session, processing_day=first.processing_day) == 1
    passages = list(session.scalars(select(SearchPassage)))
    assert {item.daily_log_revision_id for item in passages} == {second.id, other.id}
    assert len(passages) == 2
    assert session.scalar(select(func.count()).select_from(DailyLogRevision)) == 3

    hit = session.scalars(
        select(SearchPassage).where(
            SearchPassage.search_text.bool_op("@@")(func.plainto_tsquery("simple", "completed"))
        )
    ).one()
    assert hit.details["daily_log_revision_id"] == second.id
    assert hit.embedding is None
    assert hit.embedding_config is None


def test_source_replacement_is_scoped_and_invalidates_old_embeddings(session: Session) -> None:
    sources = [
        ("discord", "chat"),
        ("discord", "other"),
        ("agent_chat_ui", "chat"),
    ]
    originals: dict[tuple[str, str], int] = {}
    for source, conversation_id in sources:
        message = capture_source_message(
            session,
            source=source,
            conversation_id=conversation_id,
            message_id="first",
            role="user",
            content="I called Harsha.",
        )
        originals[(source, conversation_id)] = message.id
        assert replace_source_passages(session, source=source, conversation_id=conversation_id) == 1

    target_id = f"source:{originals[('discord', 'chat')]}:0"
    original = session.get_one(SearchPassage, target_id)
    original.embedding = [1.0] + [0.0] * 1023
    original.embedding_config = {"model": "synthetic", "dimensions": 1024}
    session.flush()
    untouched = {
        item.id: item.content
        for item in session.scalars(select(SearchPassage).where(SearchPassage.id != target_id))
    }

    later = capture_source_message(
        session,
        source="discord",
        conversation_id="chat",
        message_id="second",
        role="assistant",
        content="How did the call go?",
    )
    for _ in range(2):
        assert replace_source_passages(session, source="discord", conversation_id="chat") == 1

    rebuilt = session.get_one(SearchPassage, target_id)
    assert rebuilt.details["source_message_ids"] == [originals[("discord", "chat")], later.id]
    assert "user: I called Harsha." in rebuilt.content
    assert "assistant: How did the call go?" in rebuilt.content
    assert rebuilt.embedding is None
    assert rebuilt.embedding_config is None
    assert {
        item.id: item.content
        for item in session.scalars(select(SearchPassage).where(SearchPassage.id != target_id))
    } == untouched
    assert session.scalar(select(func.count()).select_from(SearchPassage)) == 3


def test_lexical_search_excludes_stale_logs_before_reindexing(session: Session) -> None:
    first = DailyLogRevision(
        processing_day=date(2026, 10, 1),
        revision_number=1,
        content="## Meals\nLunch at home.",
        as_of=datetime.fromisoformat("2026-10-01T20:00:00+05:30"),
    )
    session.add(first)
    session.flush()
    replace_daily_log_passages(session, processing_day=first.processing_day)

    source = capture_source_message(
        session,
        source="discord",
        conversation_id="chat",
        message_id="lunch",
        role="user",
        content="Lunch was good.",
    )
    replace_source_passages(session, source="discord", conversation_id="chat")
    source_passage_id = f"source:{source.id}:0"

    hits = search_lexical_passages(session, query="lunch")
    assert {item.id for item in hits} == {f"daily-log:{first.id}:0", source_passage_id}
    assert all(item.embedding is None for item in hits)

    second = DailyLogRevision(
        processing_day=first.processing_day,
        revision_number=2,
        content="## Meals\nLunch at the office.",
        as_of=first.as_of,
    )
    session.add(second)
    session.flush()

    # The old passage still exists, but its canonical revision is no longer current.
    assert session.get(SearchPassage, f"daily-log:{first.id}:0") is not None
    hits = search_lexical_passages(session, query="lunch")
    assert [item.id for item in hits] == [source_passage_id]

    replace_daily_log_passages(session, processing_day=first.processing_day)
    hits = search_lexical_passages(session, query="lunch")
    assert {item.id for item in hits} == {f"daily-log:{second.id}:0", source_passage_id}
    assert search_lexical_passages(session, query="home") == []


def test_embeddings_round_trip_without_repeating_model_calls(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_revision = DailyLogRevision(
        processing_day=date(2026, 10, 1),
        revision_number=1,
        content="## Tasks\nCall completed.\n\n## Meals\nLunch at home.",
        as_of=datetime.fromisoformat("2026-10-01T20:00:00+05:30"),
    )
    session.add(log_revision)
    session.flush()
    replace_daily_log_passages(session, processing_day=log_revision.processing_day)

    calls: list[list[str]] = []

    def embed_documents(_self: OllamaEmbeddings, texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(len(text))] + [0.0] * 1023 for text in texts]

    monkeypatch.setattr(OllamaEmbeddings, "embed_documents", embed_documents)
    model = OllamaEmbeddings(model="test")
    embedding_config: dict[str, object] = {"model": "test", "dimensions": 1024}

    assert (
        embed_daily_log_passages(
            session,
            daily_log_revision_id=log_revision.id,
            model=model,
            embedding_config=embedding_config,
        )
        == 2
    )
    session.expire_all()

    for passage in session.scalars(select(SearchPassage)):
        assert passage.embedding is not None
        assert passage.embedding[0] == float(len(passage.content))
        assert passage.embedding_config == embedding_config

    assert (
        embed_daily_log_passages(
            session,
            daily_log_revision_id=log_revision.id,
            model=model,
            embedding_config=embedding_config,
        )
        == 0
    )
    assert len(calls) == 1


def test_semantic_search_ranks_only_current_compatible_passages(session: Session) -> None:
    processing_day = date(2026, 10, 1)
    as_of = datetime.fromisoformat("2026-10-01T20:00:00+05:30")
    older = DailyLogRevision(
        processing_day=processing_day,
        revision_number=1,
        content="Earlier account.",
        as_of=as_of,
    )
    latest = DailyLogRevision(
        processing_day=processing_day,
        revision_number=2,
        content="Current account.",
        as_of=as_of,
    )
    session.add_all([older, latest])
    session.flush()

    padding = [0.0] * 1022
    query_embedding = [1.0, 0.0, *padding]
    config: dict[str, object] = {"model": "test", "digest": "current", "dimensions": 1024}
    for passage_id, revision_id, vector, passage_config in [
        ("stale", older.id, query_embedding, config),
        ("incompatible", latest.id, query_embedding, {**config, "digest": "different"}),
        ("nearer", latest.id, [0.8, 0.2, *padding], config),
        ("farther", latest.id, [0.0, 1.0, *padding], config),
    ]:
        session.add(
            SearchPassage(
                id=passage_id,
                daily_log_revision_id=revision_id,
                content=passage_id,
                details={},
                embedding=vector,
                embedding_config=passage_config,
            )
        )
    session.flush()

    hits = search_semantic_passages(
        session, query_embedding=query_embedding, embedding_config=config, limit=2
    )
    assert [hit.id for hit in hits] == ["nearer", "farther"]


def test_hybrid_search_combines_rankings_without_losing_lexical_only_evidence(
    session: Session,
) -> None:
    log_revision = DailyLogRevision(
        processing_day=date(2026, 10, 1),
        revision_number=1,
        content="Phone purchase research.",
        as_of=datetime.fromisoformat("2026-10-01T20:00:00+05:30"),
    )
    session.add(log_revision)
    session.flush()

    padding = [0.0] * 1022
    query_embedding = [1.0, 0.0, *padding]
    config: dict[str, object] = {"model": "test", "dimensions": 1024}
    for passage_id, content, vector in [
        ("lexical-only", "phone", None),
        ("semantic-only", "A handset purchase.", query_embedding),
        ("shared", "phone", [0.8, 0.2, *padding]),
    ]:
        session.add(
            SearchPassage(
                id=passage_id,
                daily_log_revision_id=log_revision.id,
                content=content,
                details={},
                embedding=vector,
                embedding_config=config if vector is not None else None,
            )
        )
    session.flush()

    hits = search_hybrid_passages(
        session,
        query="phone",
        query_embedding=query_embedding,
        embedding_config=config,
        limit=3,
        candidate_limit=3,
    )
    assert [hit.id for hit in hits] == ["shared", "lexical-only", "semantic-only"]


def test_source_refresh_preserves_unchanged_vectors_and_removes_obsolete_passages(
    session: Session,
) -> None:
    first = capture_source_message(
        session,
        source="discord",
        conversation_id="chat",
        message_id="first",
        role="user",
        content="Earlier discussion. " * 130,
    )
    tail = capture_source_message(
        session,
        source="discord",
        conversation_id="chat",
        message_id="tail",
        role="assistant",
        content="Current discussion. " * 40,
    )
    assert replace_source_passages(session, source="discord", conversation_id="chat") == 2
    first_id = f"source:{first.id}:0"
    tail_id = f"source:{tail.id}:1"
    config: dict[str, object] = {"model": "synthetic", "dimensions": 1024}
    for passage_id in (first_id, tail_id):
        passage = session.get_one(SearchPassage, passage_id)
        passage.embedding = [1.0] + [0.0] * 1023
        passage.embedding_config = config.copy()
    session.flush()

    capture_source_message(
        session,
        source="discord",
        conversation_id="chat",
        message_id="later",
        role="user",
        content="One more detail.",
    )
    replace_source_passages(session, source="discord", conversation_id="chat")
    session.expire_all()
    unchanged = session.get_one(SearchPassage, first_id)
    changed = session.get_one(SearchPassage, tail_id)
    assert unchanged.embedding is not None and unchanged.embedding[0] == 1.0
    assert unchanged.embedding_config == config
    assert changed.embedding is None and changed.embedding_config is None

    changed.embedding = [2.0] + [0.0] * 1023
    changed.embedding_config = config.copy()
    session.flush()
    replace_source_passages(session, source="discord", conversation_id="chat")
    session.expire_all()
    repeated = session.get_one(SearchPassage, tail_id)
    assert repeated.embedding is not None and repeated.embedding[0] == 2.0
    assert repeated.embedding_config == config

    session.delete(tail)
    assert replace_source_passages(session, source="discord", conversation_id="chat") == 1
    session.expire_all()
    assert session.get(SearchPassage, tail_id) is None


def test_source_read_pages_preserve_original_text_and_attribution(session: Session) -> None:
    message = capture_source_message(
        session,
        source="agent_chat_ui",
        conversation_id="phone",
        message_id="friend",
        role="user",
        content="My friend works at Accenture. " * 140,
    )
    message.captured_at = datetime.fromisoformat("2026-10-01T21:59:00+05:30")
    session.flush()
    session.expire_all()

    first = read_source_message_page(session, source_message_id=message.id)
    assert first.metadata["next_offset"] == 3000
    second = read_source_message_page(session, source_message_id=message.id, offset=3000)
    assert second.metadata["next_offset"] is None
    assert first.page_content.split("\n\n", 1)[1] == message.content[:3000]
    assert second.page_content.split("\n\n", 1)[1] == message.content[3000:]
    assert "Thursday, 01 October 2026, 21:59 IST] user:" in first.page_content
    assert first.metadata["source_message_id"] == message.id
    assert first.metadata["conversation_id"] == "phone"
    with pytest.raises(ValueError, match="limit"):
        read_source_message_page(session, source_message_id=message.id, limit=3001)


def test_log_read_pages_reject_a_superseded_revision(session: Session) -> None:
    original = DailyLogRevision(
        processing_day=date(2026, 10, 1),
        revision_number=1,
        content="## Phone\nBorrowed a phone. " * 150,
        as_of=datetime.fromisoformat("2026-10-01T20:00:00+05:30"),
    )
    session.add(original)
    session.flush()
    page = read_daily_log_page(session, daily_log_revision_id=original.id)
    assert page.page_content.split("\n\n", 1)[1] == original.content[:3000]
    assert page.metadata["next_offset"] == 3000

    latest = DailyLogRevision(
        processing_day=original.processing_day,
        revision_number=2,
        content="## Phone\nReturned the borrowed phone.",
        as_of=original.as_of,
    )
    session.add(latest)
    session.flush()
    session.expire_all()
    # Continuing a now-superseded narrative must not return old or substituted text.
    with pytest.raises(NoResultFound):
        read_daily_log_page(session, daily_log_revision_id=original.id, offset=3000)

    # Canonical reads do not depend on passage publication or embedding availability.
    current = read_daily_log_page(session, daily_log_revision_id=latest.id)
    assert current.page_content.split("\n\n", 1)[1] == latest.content
    assert current.metadata["daily_log_revision_id"] == latest.id
    assert current.metadata["revision_number"] == 2
    assert current.metadata["next_offset"] is None
