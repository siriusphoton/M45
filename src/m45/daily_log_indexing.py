from dataclasses import dataclass
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45.config import Settings
from m45.embeddings import is_transient_embedding_error, prepare_embeddings
from m45.search_index import embed_daily_log_passages


@dataclass
class DailyLogIndexingContext:
    engine: Engine
    settings: Settings


class DailyLogIndexingState(TypedDict):
    daily_log_revision_id: int
    embedded_count: NotRequired[int]


def _embed_passages(
    state: DailyLogIndexingState, runtime: Runtime[DailyLogIndexingContext]
) -> dict[str, int]:
    """Embed published passages for one saved revision, retrying trasnient failures."""
    model, embedding_config = prepare_embeddings(runtime.context.settings)
    with Session(runtime.context.engine) as session, session.begin():
        embedded_count = embed_daily_log_passages(
            session,
            daily_log_revision_id=state["daily_log_revision_id"],
            model=model,
            embedding_config=embedding_config,
        )
    return {"embedded_count": embedded_count}


_builder = StateGraph(DailyLogIndexingState, context_schema=DailyLogIndexingContext)
_builder.add_node(  # pyright: ignore[reportUnknownMemberType]
    "embed_passages",
    _embed_passages,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        retry_on=is_transient_embedding_error,
    ),
)
_builder.add_edge(START, "embed_passages")
_builder.add_edge("embed_passages", END)
embed_saved_daily_log = _builder.compile()  # pyright: ignore[reportUnknownMemberType]
