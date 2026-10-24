from datetime import UTC, date, datetime
from typing import Literal, cast
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Select,
    Text,
    delete,
    func,
    literal,
    or_,
    select,
    tuple_,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, Session, aliased, mapped_column

from m45.daily_log import (
    DailyLogRevision,
    load_latest_daily_log_revision,
)
from m45.database import Base
from m45.indexing import build_daily_log_passages, build_source_passages
from m45.source_history import SourceMessage

RRF_RANK_CONSTANT = 60
READ_PAGE_CHARACTERS = 3_000
type SearchScope = Literal["all", "daily_log", "source_messages"]

BROWSE_PAGE_SIZE = 10
BROWSE_PREVIEW_CHARACTERS = 300

type BrowseDirection = Literal["before", "after"]


def _select_latest_daily_log_revision_ids() -> Select[tuple[int]]:
    """Build a query selecting the latest saved revision ID for each processing day."""
    newer = aliased(DailyLogRevision)
    return select(DailyLogRevision.id).where(
        ~select(newer.id)
        .where(
            newer.processing_day == DailyLogRevision.processing_day,
            newer.revision_number > DailyLogRevision.revision_number,
        )
        .exists()
    )


class SearchPassage(Base):
    """A rebuildable search copy of a log section or source-conversation excerpt."""

    __tablename__ = "search_passages"
    __table_args__ = (
        CheckConstraint(
            "(daily_log_revision_id IS NOT NULL AND source IS NULL "
            "AND conversation_id IS NULL) OR "
            "(daily_log_revision_id IS NULL AND source IS NOT NULL "
            "AND conversation_id IS NOT NULL)",
            name="ck_search_passages_parent",
        ),
        CheckConstraint("content <> ''", name="ck_search_passages_content_nonempty"),
        CheckConstraint(
            "(embedding IS NULL) = (embedding_config IS NULL)",
            name="ck_search_passages_embedding_config",
        ),
        Index("ix_search_passages_log", "daily_log_revision_id"),
        Index("ix_search_passages_conversation", "source", "conversation_id"),
        Index("ix_search_passages_lexical", "search_text", postgresql_using="gin"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    daily_log_revision_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("daily_log_revisions.id")
    )
    source: Mapped[str | None] = mapped_column(Text)
    conversation_id: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, object]] = mapped_column(JSONB)
    search_text: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple'::regconfig, content)", persisted=True),
    )
    embedding: Mapped[list[float] | None] = mapped_column(VECTOR(1024))
    embedding_config: Mapped[dict[str, object] | None] = mapped_column(JSONB(none_as_null=True))
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def replace_daily_log_passages(session: Session, *, processing_day: date) -> int:
    """Replace a day's passages from its latest log revision; caller owns commit."""
    log_revision = load_latest_daily_log_revision(session, processing_day=processing_day)
    if log_revision is None:
        raise ValueError(f"no daily log exists for {processing_day}")
    passages = build_daily_log_passages(log_revision)

    session.execute(
        delete(SearchPassage).where(
            SearchPassage.daily_log_revision_id.in_(
                select(DailyLogRevision.id).where(DailyLogRevision.processing_day == processing_day)
            )
        )
    )
    session.add_all(
        SearchPassage(
            id=passage.id,
            daily_log_revision_id=log_revision.id,
            content=passage.page_content,
            details=passage.metadata,
        )
        for passage in passages
    )
    session.flush()
    return len(passages)


def replace_source_passages(session: Session, *, source: str, conversation_id: str) -> int:
    """Refresh a conversation's passages, keeping unchanged vectors; caller owns commit."""
    messages = list(
        session.scalars(
            select(SourceMessage).where(
                SourceMessage.source == source,
                SourceMessage.conversation_id == conversation_id,
            )
        )
    )
    passages = build_source_passages(messages)

    existing = {
        passage.id: passage
        for passage in session.scalars(
            select(SearchPassage).where(
                SearchPassage.source == source,
                SearchPassage.conversation_id == conversation_id,
            )
        )
    }
    for passage in passages:
        stored = existing.pop(cast(str, passage.id), None)
        if stored is None:
            session.add(
                SearchPassage(
                    id=passage.id,
                    source=source,
                    conversation_id=conversation_id,
                    content=passage.page_content,
                    details=passage.metadata,
                )
            )
        else:
            if stored.content != passage.page_content:
                stored.content = passage.page_content
                stored.embedding = None
                stored.embedding_config = None
                stored.indexed_at = datetime.now(UTC)
            stored.details = passage.metadata

    for obsolete in existing.values():
        session.delete(obsolete)
    session.flush()
    return len(passages)


def search_lexical_passages(
    session: Session, *, query: str, scope: SearchScope = "all", limit: int = 10
) -> list[SearchPassage]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    terms = func.plainto_tsquery("simple", query)
    rank = func.ts_rank_cd(SearchPassage.search_text, terms)
    statement = (
        select(SearchPassage)
        .where(
            SearchPassage.search_text.bool_op("@@")(terms),
            or_(
                SearchPassage.daily_log_revision_id.is_(None),
                SearchPassage.daily_log_revision_id.in_(_select_latest_daily_log_revision_ids()),
            ),
        )
        .order_by(rank.desc(), SearchPassage.id)
        .limit(limit)
    )

    if scope == "daily_log":
        statement = statement.where(SearchPassage.daily_log_revision_id.is_not(None))
    elif scope == "source_messages":
        statement = statement.where(SearchPassage.daily_log_revision_id.is_(None))
    return list(session.scalars(statement))


def search_semantic_passages(
    session: Session,
    *,
    query_embedding: list[float],
    embedding_config: dict[str, object],
    scope: SearchScope = "all",
    limit: int = 10,
) -> list[SearchPassage]:
    """Rank eligible, compatible passages by cosine distance from the query."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    distance = SearchPassage.embedding.cosine_distance(query_embedding)
    statement = (
        select(SearchPassage)
        .where(
            SearchPassage.embedding.is_not(None),
            SearchPassage.embedding_config == embedding_config,
            or_(
                SearchPassage.daily_log_revision_id.is_(None),
                SearchPassage.daily_log_revision_id.in_(_select_latest_daily_log_revision_ids()),
            ),
        )
        .order_by(distance, SearchPassage.id)
        .limit(limit)
    )

    if scope == "daily_log":
        statement = statement.where(SearchPassage.daily_log_revision_id.is_not(None))
    elif scope == "source_messages":
        statement = statement.where(SearchPassage.daily_log_revision_id.is_(None))
    return list(session.scalars(statement))


def embed_daily_log_passages(
    session: Session,
    *,
    daily_log_revision_id: int,
    model: OllamaEmbeddings,
    embedding_config: dict[str, object],
) -> int:
    """Fill missing vectors for one indexed log revision; caller owns commit."""
    passages = list(
        session.scalars(
            select(SearchPassage)
            .where(SearchPassage.daily_log_revision_id == daily_log_revision_id)
            .order_by(SearchPassage.id)
        )
    )
    if not passages:
        raise ValueError(f"No indexed passages for daily-log revision {daily_log_revision_id}")

    return _embed_missing_passages(
        session, passages=passages, model=model, embedding_config=embedding_config
    )


def embed_source_passages(
    session: Session,
    *,
    source: str,
    conversation_id: str,
    model: OllamaEmbeddings,
    embedding_config: dict[str, object],
) -> int:
    """Fill missing vectors for one indexed source conversation; caller owns commit."""
    passages = list(
        session.scalars(
            select(SearchPassage)
            .where(
                SearchPassage.source == source,
                SearchPassage.conversation_id == conversation_id,
            )
            .order_by(SearchPassage.id)
        )
    )
    if not passages:
        raise ValueError(f"No indexed passages for source conversation {source}/{conversation_id}")

    return _embed_missing_passages(
        session, passages=passages, model=model, embedding_config=embedding_config
    )


def _embed_missing_passages(
    session: Session,
    *,
    passages: list[SearchPassage],
    model: OllamaEmbeddings,
    embedding_config: dict[str, object],
) -> int:
    """Embed only pending passage rows and flush their vectors; caller owns commit."""
    pending = [passage for passage in passages if passage.embedding is None]
    if not pending:
        return 0

    vectors = model.embed_documents([passage.content for passage in pending])
    pairs = list(zip(pending, vectors, strict=True))
    for passage, vector in pairs:
        passage.embedding = vector
        passage.embedding_config = embedding_config.copy()

    session.flush()
    return len(pending)


def search_hybrid_passages(
    session: Session,
    *,
    query: str,
    query_embedding: list[float],
    embedding_config: dict[str, object],
    scope: SearchScope = "all",
    limit: int = 10,
    candidate_limit: int = 20,
) -> list[SearchPassage]:
    """Combine lexical and semantic candidates using reciprocal rank fusion."""
    if not 0 < limit <= candidate_limit:
        raise ValueError("limits must satisfy 0 < limit <= candidate_limit")

    lexical = search_lexical_passages(session, query=query, scope=scope, limit=candidate_limit)
    semantic = search_semantic_passages(
        session,
        query_embedding=query_embedding,
        embedding_config=embedding_config,
        scope=scope,
        limit=candidate_limit,
    )

    passages: dict[str, SearchPassage] = {}
    scores: dict[str, float] = {}
    for ranking in (lexical, semantic):
        for rank, passage in enumerate(ranking, start=1):
            passages[passage.id] = passage
            scores[passage.id] = scores.get(passage.id, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)
    ranked_ids = sorted(scores, key=lambda passage_id: (-scores[passage_id], passage_id))
    return [passages[passage_id] for passage_id in ranked_ids[:limit]]


def _read_text_page(content: str, *, offset: int, limit: int) -> tuple[str, int | None]:
    """Slice original text and return the next offset, or None at the end."""
    if not 0 < limit <= READ_PAGE_CHARACTERS:
        raise ValueError(f"limit must be between 1 and {READ_PAGE_CHARACTERS}")
    if not 0 <= offset < len(content):
        raise ValueError("offset must point inside the content")
    end = min(offset + limit, len(content))
    return content[offset:end], end if end < len(content) else None


def read_source_message_page(
    session: Session,
    *,
    source_message_id: int,
    offset: int = 0,
    limit: int = READ_PAGE_CHARACTERS,
) -> Document:
    """Read a bounded part of one original message, retaining its attribution."""
    message = session.get_one(SourceMessage, source_message_id)
    text, next_offset = _read_text_page(message.content, offset=offset, limit=limit)
    timestamp = message.captured_at.astimezone(ZoneInfo("Asia/Kolkata"))
    return Document(
        id=f"source-message:{message.id}:{offset}",
        page_content=(
            f"Source message ({message.source})\n"
            f"[{timestamp:%A, %d %B %Y, %H:%M IST}] {message.role}:\n\n{text}"
        ),
        metadata={
            "kind": "source_message",
            "source_message_id": message.id,
            "source": message.source,
            "conversation_id": message.conversation_id,
            "role": message.role,
            "captured_at": message.captured_at.isoformat(),
            "offset": offset,
            "next_offset": next_offset,
        },
    )


def read_daily_log_page(
    session: Session,
    *,
    daily_log_revision_id: int,
    offset: int = 0,
    limit: int = READ_PAGE_CHARACTERS,
) -> Document:
    """Read a bounded part of a selected log only while its revision is current."""
    log_revision = session.scalars(
        select(DailyLogRevision).where(
            DailyLogRevision.id == daily_log_revision_id,
            DailyLogRevision.id.in_(_select_latest_daily_log_revision_ids()),
        )
    ).one()
    text, next_offset = _read_text_page(log_revision.content, offset=offset, limit=limit)
    return Document(
        id=f"daily-log-page:{log_revision.id}:{offset}",
        page_content=(
            f"Daily log for processing day {log_revision.processing_day.isoformat()}\n\n{text}"
        ),
        metadata={
            "kind": "daily_log",
            "daily_log_revision_id": log_revision.id,
            "processing_day": log_revision.processing_day.isoformat(),
            "revision_number": log_revision.revision_number,
            "offset": offset,
            "next_offset": next_offset,
        },
    )


def list_daily_logs(session: Session, *, start_day: date, end_day: date) -> dict[str, object]:
    """List current log previews in an inclusive processing-day range."""
    if end_day < start_day:
        raise ValueError("end_day must be on or after start_day")
    rows = list(
        session.scalars(
            select(DailyLogRevision)
            .where(
                DailyLogRevision.id.in_(_select_latest_daily_log_revision_ids()),
                DailyLogRevision.processing_day >= start_day,
                DailyLogRevision.processing_day <= end_day,
            )
            .order_by(DailyLogRevision.processing_day)
            .limit(BROWSE_PAGE_SIZE + 1)
        )
    )
    return {
        "logs": [
            {
                "daily_log_revision_id": row.id,
                "processing_day": row.processing_day.isoformat(),
                "revision_number": row.revision_number,
                "as_of": row.as_of.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(),
                "preview": row.content[:BROWSE_PREVIEW_CHARACTERS],
                "truncated": len(row.content) > BROWSE_PREVIEW_CHARACTERS,
            }
            for row in rows[:BROWSE_PAGE_SIZE]
        ],
        "next_start_day": (
            rows[BROWSE_PAGE_SIZE].processing_day.isoformat()
            if len(rows) > BROWSE_PAGE_SIZE
            else None
        ),
    }


def list_source_neighbors(
    session: Session, *, source_message_id: int, direction: BrowseDirection
) -> dict[str, object]:
    """List neighboring message previews within the anchor's source conversation."""
    anchor = session.get_one(SourceMessage, source_message_id)
    position = tuple_(SourceMessage.captured_at, SourceMessage.id)
    anchor_position = tuple_(literal(anchor.captured_at), literal(anchor.id))
    statement = select(SourceMessage).where(
        SourceMessage.source == anchor.source,
        SourceMessage.conversation_id == anchor.conversation_id,
    )
    if direction == "before":
        statement = statement.where(position < anchor_position).order_by(
            SourceMessage.captured_at.desc(), SourceMessage.id.desc()
        )
    else:
        statement = statement.where(position > anchor_position).order_by(
            SourceMessage.captured_at, SourceMessage.id
        )
    rows = list(session.scalars(statement.limit(BROWSE_PAGE_SIZE + 1)))
    selected = rows[:BROWSE_PAGE_SIZE]
    next_anchor_id = selected[-1].id if len(rows) > BROWSE_PAGE_SIZE else None
    if direction == "before":
        selected.reverse()
    return {
        "source": anchor.source,
        "conversation_id": anchor.conversation_id,
        "anchor_source_message_id": anchor.id,
        "direction": direction,
        "messages": [
            {
                "source_message_id": row.id,
                "role": row.role,
                "captured_at": row.captured_at.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(),
                "preview": row.content[:BROWSE_PREVIEW_CHARACTERS],
                "truncated": len(row.content) > BROWSE_PREVIEW_CHARACTERS,
            }
            for row in selected
        ],
        "next_anchor_id": next_anchor_id,
    }
