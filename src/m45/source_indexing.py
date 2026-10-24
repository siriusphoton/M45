from dataclasses import dataclass
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45.config import Settings
from m45.embeddings import is_transient_embedding_error, prepare_embeddings
from m45.search_index import embed_source_passages, replace_source_passages


@dataclass
class SourceIndexingContext:
    engine: Engine
    settings: Settings


class SourceIndexingState(TypedDict):
    source: str
    conversation_id: str
    passage_count: NotRequired[int]
    embedded_count: NotRequired[int]


def _refresh_passages(
    state: SourceIndexingState, runtime: Runtime[SourceIndexingContext]
) -> dict[str, int]:
    """Commit searchable source text before attempting embeddings."""
    with Session(runtime.context.engine) as session, session.begin():
        passage_count = replace_source_passages(
            session,
            source=state["source"],
            conversation_id=state["conversation_id"],
        )
    return {"passage_count": passage_count}


def _embed_passages(
    state: SourceIndexingState, runtime: Runtime[SourceIndexingContext]
) -> dict[str, int]:
    """Fill missing vectors in a fresh transaction on every attempt."""
    model, embedding_config = prepare_embeddings(runtime.context.settings)
    with Session(runtime.context.engine) as session, session.begin():
        embedded_count = embed_source_passages(
            session,
            source=state["source"],
            conversation_id=state["conversation_id"],
            model=model,
            embedding_config=embedding_config,
        )
    return {"embedded_count": embedded_count}


_builder = StateGraph(SourceIndexingState, context_schema=SourceIndexingContext)
_builder.add_node("refresh_passages", _refresh_passages)  # pyright: ignore[reportUnknownMemberType]
_builder.add_node(  # pyright: ignore[reportUnknownMemberType]
    "embed_passages",
    _embed_passages,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        retry_on=is_transient_embedding_error,
    ),
)
_builder.add_edge(START, "refresh_passages")
_builder.add_edge("refresh_passages", "embed_passages")
_builder.add_edge("embed_passages", END)
index_source_conversation = _builder.compile()  # pyright: ignore[reportUnknownMemberType]
