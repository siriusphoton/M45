from datetime import date, datetime

import pytest
from langchain_ollama import OllamaEmbeddings
from ollama import ResponseError
from sqlalchemy import delete, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45 import daily_log_indexing
from m45.config import Settings, load_settings
from m45.daily_log import DailyLogRevision
from m45.daily_log_indexing import DailyLogIndexingContext, embed_saved_daily_log
from m45.search_index import SearchPassage, replace_daily_log_passages, search_lexical_passages


def test_embedding_outage_preserves_saved_log_and_allows_recovery(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = OllamaEmbeddings(model="test")

    def prepare_embeddings(_settings: Settings) -> tuple[OllamaEmbeddings, dict[str, object]]:
        return model, {"model": "test", "dimensions": 1024}

    attempts = 0
    available = False

    def embed_documents(_self: OllamaEmbeddings, texts: list[str]) -> list[list[float]]:
        nonlocal attempts
        attempts += 1
        if not available:
            raise ResponseError("temporarily unavailable", status_code=503)
        return [[1.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setattr(daily_log_indexing, "prepare_embeddings", prepare_embeddings)
    monkeypatch.setattr(OllamaEmbeddings, "embed_documents", embed_documents)
    context = DailyLogIndexingContext(engine=engine, settings=load_settings())
    processing_day = date(2026, 10, 1)

    with Session(engine) as session, session.begin():
        revision = DailyLogRevision(
            processing_day=processing_day,
            revision_number=1,
            content="## Meals\nLunch at home.",
            as_of=datetime.fromisoformat("2026-10-01T20:00:00+05:30"),
        )
        session.add(revision)
        session.flush()
        revision_id = revision.id
        replace_daily_log_passages(session, processing_day=processing_day)

    try:
        with pytest.raises(ResponseError, match="temporarily unavailable"):
            embed_saved_daily_log.invoke(  # pyright: ignore[reportUnknownMemberType]
                {"daily_log_revision_id": revision_id}, context=context
            )
        assert attempts == 3

        with Session(engine) as session:
            hits = search_lexical_passages(session, query="lunch")
            assert len(hits) == 1
            assert hits[0].embedding is None
            assert session.get(DailyLogRevision, revision_id) is not None

        available = True
        result = embed_saved_daily_log.invoke(  # pyright: ignore[reportUnknownMemberType]
            {"daily_log_revision_id": revision_id}, context=context
        )
        assert result["embedded_count"] == 1
        with Session(engine) as session:
            passage = session.scalars(select(SearchPassage)).one()
            assert passage.embedding is not None
            assert passage.daily_log_revision_id == revision_id
            assert session.scalar(select(func.count()).select_from(DailyLogRevision)) == 1
    finally:
        # The workflow commits its own sessions, outside the usual rollback fixture.
        with Session(engine) as session, session.begin():
            session.execute(
                delete(SearchPassage).where(SearchPassage.daily_log_revision_id == revision_id)
            )
            session.execute(delete(DailyLogRevision).where(DailyLogRevision.id == revision_id))
