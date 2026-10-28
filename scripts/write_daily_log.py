import argparse
import asyncio
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from m45.agent import create_configured_chat_model
from m45.config import load_settings
from m45.daily_log import (
    append_daily_log_revision,
    draft_daily_log,
    load_daily_source_messages,
    load_latest_daily_log_revision,
    processing_day_at,
)
from m45.daily_log_indexing import DailyLogIndexingContext, embed_saved_daily_log
from m45.database import create_database_engine
from m45.interaction import AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE
from m45.search_index import replace_daily_log_passages


async def write_daily_log(
    processing_day: date,
    as_of: datetime,
    *,
    expected_database: Literal["m45", "m45_trial"],
) -> None:
    """Generate, save and index one log in the explicitly selected database."""
    settings = load_settings()
    engine = create_database_engine(settings)
    try:
        if engine.url.database != expected_database or engine.url.username != expected_database:
            raise ValueError(f"Use the {expected_database} database and login")

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
        if not messages:
            print(f"No source messages for {processing_day}; no log written.")
            return
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
            f"Saved {expected_database} revision {revision_number}, "
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


async def main() -> None:
    """Write the previous and current processing days using one real cutoff."""
    as_of = datetime.now(ZoneInfo("Asia/Kolkata"))
    current_day = processing_day_at(as_of)
    for day in (current_day - timedelta(days=1), current_day):
        await write_daily_log(day, as_of, expected_database="m45")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write live daily logs for the previous and current processing days."
    )
    parser.parse_args()
    asyncio.run(main())
