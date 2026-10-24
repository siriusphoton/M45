from collections.abc import Sequence
from itertools import groupby
from zoneinfo import ZoneInfo

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from m45.daily_log import DailyLogRevision
from m45.source_history import SourceMessage


def build_daily_log_passages(
    log_revision: DailyLogRevision,
    *,
    chunk_size: int = 3_000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Prepare searchable passages from one saved log revision without writing it."""
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")],
        strip_headers=True,
    ).split_text(log_revision.content)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(sections)

    passages: list[Document] = []
    for position, chunk in enumerate(chunks):
        section = chunk.metadata.get("section", "Overview")
        passages.append(
            Document(
                id=f"daily-log:{log_revision.id}:{position}",
                page_content=(
                    f"Daily log for processing day {log_revision.processing_day.isoformat()}\n"
                    f"Section: {section}\n\n"
                    f"{chunk.page_content}"
                ),
                metadata={
                    "kind": "daily_log",
                    "daily_log_revision_id": log_revision.id,
                    "processing_day": log_revision.processing_day.isoformat(),
                    "revision_number": log_revision.revision_number,
                    "section": section,
                    "chunk_index": position,
                },
            )
        )

    return passages


def build_source_passages(
    messages: Sequence[SourceMessage],
    *,
    chunk_size: int = 3_000,
) -> list[Document]:
    """Build bounded conversation excerpts with exact source-row references."""
    ordered = sorted(
        messages,
        key=lambda message: (
            message.source,
            message.conversation_id,
            message.captured_at,
            message.id,
        ),
    )
    passages: list[Document] = []

    for _, conversation in groupby(
        ordered, key=lambda message: (message.source, message.conversation_id)
    ):
        passage_parts: list[list[tuple[SourceMessage, str]]] = []
        pending: list[tuple[SourceMessage, str]] = []
        used = 0

        for message in conversation:
            timestamp = message.captured_at.astimezone(ZoneInfo("Asia/Kolkata"))
            header = f"[{timestamp: %A, %d %B %Y, %H:%M:%S} IST] {message.role}: "
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size - len(header), chunk_overlap=0
            )
            for text in splitter.split_text(message.content):
                content = header + text
                if pending and used + 2 + len(content) > chunk_size:
                    passage_parts.append(pending)
                    previous = pending[-1]
                    pending = (
                        [previous] if len(previous[1]) + 2 + len(content) <= chunk_size else []
                    )
                    used = len(pending[0][1]) if pending else 0

                used += len(content) + (2 if pending else 0)
                pending.append((message, content))

        if pending:
            passage_parts.append(pending)

        passages.extend(
            _source_passage(parts, position) for position, parts in enumerate(passage_parts)
        )

    return passages


def _source_passage(parts: Sequence[tuple[SourceMessage, str]], position: int) -> Document:
    """Render one nonempty passage; its parts belong to the same conversation."""
    first = parts[0][0]
    last = parts[-1][0]

    return Document(
        id=f"source:{first.id}:{position}",
        page_content=(
            f"Source conversation ({first.source})\nMessage timestamps:\n\n"
            + "\n\n".join(content for _, content in parts)
        ),
        metadata={
            "kind": "source_messages",
            "source": first.source,
            "conversation_id": first.conversation_id,
            "source_message_ids": list(dict.fromkeys(message.id for message, _ in parts)),
            "captured_from": first.captured_at.isoformat(),
            "captured_to": last.captured_at.isoformat(),
            "chunk_index": position,
        },
    )
