from sqlalchemy import select
from sqlalchemy.orm import Session

from m45.config import load_settings
from m45.database import create_database_engine
from m45.interaction import AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE
from m45.source_history import SourceMessage
from m45.source_indexing import SourceIndexingContext, index_source_conversation


def main() -> None:
    """Refresh and embed retained conversations in the live application database."""
    settings = load_settings()
    engine = create_database_engine(settings)
    try:
        if engine.url.database != "m45" or engine.url.username != "m45":
            raise ValueError("Use the m45 database and login for live source indexing")

        with Session(engine) as session:
            conversations = session.execute(
                select(SourceMessage.source, SourceMessage.conversation_id)
                .where(SourceMessage.source.in_((AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE)))
                .distinct()
                .order_by(SourceMessage.source, SourceMessage.conversation_id)
            ).all()

        print(f"Found {len(conversations)} retained conversations.")
        context = SourceIndexingContext(engine=engine, settings=settings)
        for source, conversation_id in conversations:
            print(f"Indexing {source}/{conversation_id}")

            result = index_source_conversation.invoke(  # pyright: ignore[reportUnknownMemberType]
                {"source": source, "conversation_id": conversation_id}, context=context
            )
            print(
                f"{result['passage_count']} passages; "
                f"{result['embedded_count']} missing vectors filled."
            )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
