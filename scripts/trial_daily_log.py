import argparse
import asyncio
from datetime import date, datetime

from sqlalchemy.orm import Session

from m45.agent import create_configured_chat_model
from m45.config import load_settings
from m45.daily_log import (
    append_daily_log_revision,
    draft_daily_log,
    load_daily_source_messages,
    load_latest_daily_log_revision,
)
from m45.daily_log_indexing import DailyLogIndexingContext, embed_saved_daily_log
from m45.database import create_database_engine
from m45.interaction import AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE
from m45.search_index import replace_daily_log_passages


async def main(processing_day: date, as_of: datetime) -> None:
    """Generate and save one daily-log revision in the isolated trial database."""
    settings = load_settings()
    engine = create_database_engine(settings)
    try:
        if engine.url.database != "m45_trial" or engine.url.username != "m45_trial":
            raise ValueError("Use the m45_trial database and login for historial trials")

        with Session(engine) as session:
            latest_log_revision = load_latest_daily_log_revision(
                session, processing_day=processing_day
            )
            base_revision_number = (
                latest_log_revision.revision_number if latest_log_revision is not None else 0
            )
            previous_log = latest_log_revision.content if latest_log_revision is not None else None
            messages = load_daily_source_messages(
                session,
                processing_day=processing_day,
                eligible_sources=(AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE),
                before=as_of,
            )
            source_ids = {message.id for message in messages}

        print(
            f"Drafting {processing_day} from {len(messages)} sources; "
            f"base revision {base_revision_number}"
        )
        draft = await draft_daily_log(
            create_configured_chat_model(settings),
            processing_day=processing_day,
            as_of=as_of,
            messages=messages,
            previous_log=previous_log,
        )

        with Session(engine) as session, session.begin():
            saved_log_revision = append_daily_log_revision(
                session,
                processing_day=processing_day,
                base_revision_number=base_revision_number,
                content=draft,
                as_of=as_of,
                source_message_ids=source_ids,
            )
            revision_number = saved_log_revision.revision_number
            daily_log_revision_id = saved_log_revision.id
            passage_count = replace_daily_log_passages(session, processing_day=processing_day)

        print(
            f"Saved trial revision {revision_number}, "
            f"referencing {len(source_ids)} source messages."
        )
        print(draft)
        print(
            f"Published {passage_count} lexical passages for revision ID {daily_log_revision_id}."
        )

        try:
            indexing_result = await embed_saved_daily_log.ainvoke(  # pyright: ignore[reportUnknownMemberType]
                {"daily_log_revision_id": daily_log_revision_id},
                context=DailyLogIndexingContext(engine=engine, settings=settings),
            )
        except Exception as error:
            error.add_note(
                f"Embedding did not complete for revision ID {daily_log_revision_id}. "
                "The log and lexical passages are saved. Retry embedding this revision; "
                "do not rerun the writer just to recover embeddings."
            )
            raise
        print(
            f"Embedded {indexing_result['embedded_count']} passages "
            f"for revision ID {daily_log_revision_id}."
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a daily log in m45_trial.")
    parser.add_argument(
        "processing_day", type=date.fromisoformat, help="Processing day: YYYY-MM-DD"
    )
    parser.add_argument(
        "--as-of",
        type=datetime.fromisoformat,
        required=True,
        help="Exclusive review cutoff, including timezone offset",
    )
    arguments = parser.parse_args()
    asyncio.run(main(arguments.processing_day, arguments.as_of))
