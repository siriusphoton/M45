from sqlalchemy import and_, not_, select
from sqlalchemy.orm import Session

from m45.source_history import SourceMessage


def load_recent_personal_context(
    session: Session,
    *,
    eligible_sources: tuple[str, ...],
    current_source: str,
    current_conversation_id: str,
    message_limit: int,
    character_limit: int,
) -> list[SourceMessage]:
    if message_limit <= 0:
        raise ValueError("message_limit must be greater than zero")

    if character_limit <= 0:
        raise ValueError("character_limit must be greater than zero")

    if not eligible_sources:
        return []

    current_conversation = and_(
        SourceMessage.source == current_source,
        SourceMessage.conversation_id == current_conversation_id,
    )

    statement = (
        select(SourceMessage)
        .where(
            SourceMessage.source.in_(eligible_sources),
            not_(current_conversation),
        )
        .order_by(
            SourceMessage.captured_at.desc(),
            SourceMessage.id.desc(),
        )
        .limit(message_limit)
    )

    newest_first = list(session.scalars(statement))
    selected: list[SourceMessage] = []
    selected_characters = 0

    for message in newest_first:
        next_character_count = selected_characters + len(message.content)

        if next_character_count > character_limit:
            break

        selected.append(message)
        selected_characters = next_character_count

    selected.reverse()
    return selected
