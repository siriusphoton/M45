import json
from datetime import date, datetime
from typing import cast

import pytest
from langchain_core.messages import ToolMessage
from langchain_ollama import OllamaEmbeddings
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45 import retrieval_tools
from m45.config import Settings, load_settings
from m45.daily_log import DailyLogRevision
from m45.retrieval_tools import (
    create_daily_log_read_tool,
    create_memory_search_tool,
    create_source_message_read_tool,
)
from m45.search_index import SearchPassage, replace_daily_log_passages, replace_source_passages
from m45.source_history import SourceMessage, capture_source_message


def test_memory_tools_preserve_evidence_from_search_through_reading(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = OllamaEmbeddings(model="test")
    embedding_config: dict[str, object] = {"model": "test", "dimensions": 1024}
    vector = [1.0] + [0.0] * 1023
    available = True

    def prepare_embeddings(_settings: Settings) -> tuple[OllamaEmbeddings, dict[str, object]]:
        return model, embedding_config

    def embed_query(_self: OllamaEmbeddings, _text: str) -> list[float]:
        if not available:
            raise ConnectionError("embedding service unavailable")
        return vector

    monkeypatch.setattr(retrieval_tools, "prepare_embeddings", prepare_embeddings)
    monkeypatch.setattr(OllamaEmbeddings, "embed_query", embed_query)
    conversation_id = "memory-search-tool-test"
    processing_day = date(2026, 10, 3)
    try:
        with Session(engine) as session, session.begin():
            user = capture_source_message(
                session,
                source="discord",
                conversation_id=conversation_id,
                message_id="user",
                role="user",
                content="I am considering a new phone.",
            )
            assistant = capture_source_message(
                session,
                source="discord",
                conversation_id=conversation_id,
                message_id="assistant",
                role="assistant",
                content="Which phone do you prefer?",
            )
            source_ids = [user.id, assistant.id]
            log = DailyLogRevision(
                processing_day=processing_day,
                revision_number=1,
                content="## Plans\nConsidering a new phone.",
                as_of=datetime.fromisoformat("2026-10-03T20:00:00+05:30"),
            )
            session.add(log)
            session.flush()
            log_id = log.id
            replace_daily_log_passages(session, processing_day=processing_day)
            replace_source_passages(session, source="discord", conversation_id=conversation_id)
            for passage in session.scalars(
                select(SearchPassage).where(
                    (SearchPassage.conversation_id == conversation_id)
                    | (SearchPassage.daily_log_revision_id == log_id)
                )
            ):
                passage.embedding = vector
                passage.embedding_config = embedding_config

        search = create_memory_search_tool(engine, load_settings())
        source_result = search.invoke(
            {
                "name": "search_memory",
                "type": "tool_call",
                "id": "sources",
                "args": {"query": "phone", "scope": "source_messages"},
            }
        )
        assert isinstance(source_result, ToolMessage)
        assert isinstance(source_result.content, str)
        payload = cast(dict[str, object], json.loads(source_result.content))
        hits = cast(list[dict[str, object]], payload["matches"])
        assert len(hits) == 1
        metadata = cast(dict[str, object], hits[0]["metadata"])
        assert metadata["source_message_ids"] == source_ids
        selected_source_id = cast(list[int], metadata["source_message_ids"])[0]
        assert "user: I am considering" in str(hits[0]["text"])
        assert "assistant: Which phone" in str(hits[0]["text"])

        log_result = search.invoke(
            {
                "name": "search_memory",
                "type": "tool_call",
                "id": "log",
                "args": {"query": "phone", "scope": "daily_log"},
            }
        )
        assert isinstance(log_result, ToolMessage)
        assert isinstance(log_result.content, str)
        payload = cast(dict[str, object], json.loads(log_result.content))
        hits = cast(list[dict[str, object]], payload["matches"])
        assert len(hits) == 1
        metadata = cast(dict[str, object], hits[0]["metadata"])
        assert metadata["daily_log_revision_id"] == log_id
        selected_log_id = cast(int, metadata["daily_log_revision_id"])

        available = False
        with pytest.raises(ConnectionError, match="embedding service unavailable"):
            search.invoke({"query": "phone"})

        # Already discovered evidence remains readable without the embedding service.
        read_source = create_source_message_read_tool(engine)
        source_page = read_source.invoke(
            {
                "name": "read_source_message",
                "type": "tool_call",
                "id": "source-page",
                "args": {"source_message_id": selected_source_id, "offset": 5},
            }
        )
        assert isinstance(source_page, ToolMessage)
        assert isinstance(source_page.content, str)
        payload = cast(dict[str, object], json.loads(source_page.content))
        metadata = cast(dict[str, object], payload["metadata"])
        assert metadata["source_message_id"] == selected_source_id
        assert metadata["role"] == "user"
        assert metadata["offset"] == 5
        assert str(payload["text"]).endswith("considering a new phone.")

        read_log = create_daily_log_read_tool(engine)
        log_page = read_log.invoke(
            {
                "name": "read_daily_log",
                "type": "tool_call",
                "id": "log-page",
                "args": {"daily_log_revision_id": selected_log_id},
            }
        )
        assert isinstance(log_page, ToolMessage)
        assert isinstance(log_page.content, str)
        payload = cast(dict[str, object], json.loads(log_page.content))
        metadata = cast(dict[str, object], payload["metadata"])
        assert metadata["daily_log_revision_id"] == selected_log_id
        assert metadata["next_offset"] is None
        assert str(payload["text"]).endswith("## Plans\nConsidering a new phone.")
    finally:
        # The tool opens its own session, so setup must commit and clean up explicitly.
        with Session(engine) as session, session.begin():
            session.execute(
                delete(SearchPassage).where(
                    (SearchPassage.conversation_id == conversation_id)
                    | SearchPassage.daily_log_revision_id.in_(
                        select(DailyLogRevision.id).where(
                            DailyLogRevision.processing_day == processing_day
                        )
                    )
                )
            )
            session.execute(
                delete(SourceMessage).where(
                    SourceMessage.source == "discord",
                    SourceMessage.conversation_id == conversation_id,
                )
            )
            session.execute(
                delete(DailyLogRevision).where(DailyLogRevision.processing_day == processing_day)
            )
