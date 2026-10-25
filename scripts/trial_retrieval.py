import argparse
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session

from m45.agent import create_configured_chat_model
from m45.config import load_settings
from m45.daily_context import format_daily_context, load_daily_context
from m45.database import create_database_engine
from m45.interaction import AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE
from m45.retrieval import create_memory_retriever, retrieve_memory
from m45.retrieval_tools import (
    create_daily_log_browse_tool,
    create_daily_log_read_tool,
    create_memory_search_tool,
    create_source_conversation_browse_tool,
    create_source_message_read_tool,
)


async def main(question: str) -> None:
    """Inspect one fresh retrieval conversation against the isolated trial corpus."""
    settings = load_settings()
    engine = create_database_engine(settings)
    try:
        if engine.url.database != "m45_trial" or engine.url.username != "m45_trial":
            raise ValueError("Use the m45_trial database and login for retrieval trials")

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        with Session(engine) as session:
            daily_context = load_daily_context(
                session,
                now=now,
                eligible_sources=(AGENT_CHAT_UI_SOURCE, DISCORD_SOURCE),
                current_source=AGENT_CHAT_UI_SOURCE,
                current_conversation_id="retrieval-trial",
                active_message_ids=frozenset({"trial-question"}),
            )
        orientation = format_daily_context(daily_context)
        retriever = create_memory_retriever(
            create_configured_chat_model(settings),
            [
                create_memory_search_tool(engine, settings),
                create_source_message_read_tool(engine),
                create_daily_log_read_tool(engine),
                create_daily_log_browse_tool(engine),
                create_source_conversation_browse_tool(engine),
            ],
        )
        print(f"Retrieving at {now.isoformat()}; daily orientation: {len(orientation)} chars.")
        result = await retrieve_memory(
            retriever,
            messages=[HumanMessage(content=question, id="trial-question")],
            daily_context=orientation,
        )
        print(result.model_dump_json(indent=2))
    finally:
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect memory retrieval in the trial database.")
    parser.add_argument("question")
    args = parser.parse_args()
    asyncio.run(main(args.question))
