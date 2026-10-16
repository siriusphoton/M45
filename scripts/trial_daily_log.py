import argparse
import asyncio
from datetime import date, datetime

from sqlalchemy.orm import Session

from m45.agent import create_configured_chat_model
from m45.config import load_settings
from m45.daily_log import (
    draft_daily_log,
    load_current_daily_log,
    load_daily_source_messages,
    save_daily_log_revision,
)
from m45.database import create_database_engine
from m45.interaction import AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE


async def main(day: date, as_of: datetime) -> None:
    """Generate and save one daily-log revision in the isolated trial database."""
    settings = load_settings()
    engine = create_database_engine(settings)
    try:
        if engine.url.database != "m45_trial" or engine.url.username != "m45_trial":
            raise ValueError("Use the m45_trial database and login for historial trials")

        with Session(engine) as session:
            current = load_current_daily_log(session, day=day)
            expected_revision = current.revision if current is not None else 0
            previous_log = current.content if current is not None else None
            messages = load_daily_source_messages(
                session,
                day=day,
                eligible_sources=(AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE),
                before=as_of,
            )
            source_ids = {message.id for message in messages}

        print(f"Drafting {day} from {len(messages)} sources; prior revision {expected_revision}")
        draft = await draft_daily_log(
            create_configured_chat_model(settings),
            day=day,
            as_of=as_of,
            messages=messages,
            previous_log=previous_log,
        )

        with Session(engine) as session, session.begin():
            saved = save_daily_log_revision(
                session,
                day=day,
                expected_revision=expected_revision,
                content=draft,
                as_of=as_of,
                source_message_ids=source_ids,
            )
            revision = saved.revision

        print(f"Saved trial revision {revision}, referencing {len(source_ids)} source messages.")
        print(draft)
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a daily log in m45_trial.")
    parser.add_argument("day", type=date.fromisoformat, help="Processing day: YYYY-MM-DD")
    parser.add_argument(
        "--as-of",
        type=datetime.fromisoformat,
        required=True,
        help="Exclusive review cutoff, including timezone offset",
    )
    arguments = parser.parse_args()
    asyncio.run(main(arguments.day, arguments.as_of))
