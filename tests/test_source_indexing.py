import pytest
from langchain_ollama import OllamaEmbeddings
from ollama import ResponseError
from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45 import source_indexing
from m45.config import Settings, load_settings
from m45.search_index import SearchPassage, replace_source_passages, search_lexical_passages
from m45.source_history import SourceMessage, capture_source_message
from m45.source_indexing import SourceIndexingContext, index_source_conversation


def test_source_indexing_outage_keeps_text_and_recovers_only_missing_vectors(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = OllamaEmbeddings(model="test")
    embedding_config: dict[str, object] = {"model": "test", "dimensions": 1024}
    available = False
    calls: list[list[str]] = []

    def prepare_embeddings(_settings: Settings) -> tuple[OllamaEmbeddings, dict[str, object]]:
        return model, embedding_config

    def embed_documents(_self: OllamaEmbeddings, texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        if not available:
            raise ResponseError("temporarily unavailable", status_code=503)
        return [[2.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setattr(source_indexing, "prepare_embeddings", prepare_embeddings)
    monkeypatch.setattr(OllamaEmbeddings, "embed_documents", embed_documents)
    context = SourceIndexingContext(engine=engine, settings=load_settings())
    conversation_id = "source-indexing-test"
    inputs: source_indexing.SourceIndexingState = {
        "source": "discord",
        "conversation_id": conversation_id,
    }
    try:
        with Session(engine) as session, session.begin():
            first = capture_source_message(
                session,
                source="discord",
                conversation_id=conversation_id,
                message_id="first",
                role="user",
                content="Earlier discussion. " * 130,
            )
            capture_source_message(
                session,
                source="discord",
                conversation_id=conversation_id,
                message_id="tail",
                role="assistant",
                content="Current discussion. " * 40,
            )
            replace_source_passages(session, source="discord", conversation_id=conversation_id)
            first_passage_id = f"source:{first.id}:0"
            first_passage = session.get_one(SearchPassage, first_passage_id)
            first_passage.embedding = [1.0] + [0.0] * 1023
            first_passage.embedding_config = embedding_config.copy()
            capture_source_message(
                session,
                source="discord",
                conversation_id=conversation_id,
                message_id="new",
                role="user",
                content="One more detail.",
            )

        with pytest.raises(ResponseError, match="temporarily unavailable"):
            index_source_conversation.invoke(  # pyright: ignore[reportUnknownMemberType]
                inputs, context=context
            )
        assert len(calls) == 3
        with Session(engine) as session:
            hits = search_lexical_passages(session, query="detail")
            assert len(hits) == 1
            pending_id = hits[0].id
            assert hits[0].embedding is None
            preserved = session.get_one(SearchPassage, first_passage_id)
            assert preserved.embedding is not None and preserved.embedding[0] == 1.0

        available = True
        result = index_source_conversation.invoke(  # pyright: ignore[reportUnknownMemberType]
            inputs, context=context
        )
        assert result["passage_count"] == 2
        assert result["embedded_count"] == 1
        assert len(calls) == 4
        assert len(calls[-1]) == 1 and "One more detail." in calls[-1][0]
        with Session(engine) as session:
            saved = session.get_one(SearchPassage, pending_id)
            assert saved.embedding is not None and saved.embedding[0] == 2.0
            preserved = session.get_one(SearchPassage, first_passage_id)
            assert preserved.embedding is not None and preserved.embedding[0] == 1.0

        repeated = index_source_conversation.invoke(  # pyright: ignore[reportUnknownMemberType]
            inputs, context=context
        )
        assert repeated["embedded_count"] == 0
        assert len(calls) == 4
    finally:
        # Graph nodes commit independently, so the rollback session fixture cannot clean them.
        with Session(engine) as session, session.begin():
            session.execute(
                delete(SearchPassage).where(
                    SearchPassage.source == "discord",
                    SearchPassage.conversation_id == conversation_id,
                )
            )
            session.execute(
                delete(SourceMessage).where(
                    SourceMessage.source == "discord",
                    SourceMessage.conversation_id == conversation_id,
                )
            )
