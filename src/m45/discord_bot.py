import logging
from typing import Any, cast

import discord
from langchain_core.messages import (
    AIMessage,
    MessageLikeRepresentation,
    convert_to_messages,
)
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

from m45.config import load_settings
from m45.interaction import DISCORD_SOURCE, InteractionContext, discord_thread_id

DISCORD_MESSAGE_LIMIT = 2_000
TEXT_ONLY_NOTICE = "I can currently process text messages only."
PROCESSING_ERROR_NOTICE = "I couldn't process that message. Please try again."

logger = logging.getLogger(__name__)


class AgentServerResponseError(RuntimeError):
    """Raised when Agent Server returns state outside m45's output contract."""


def _assistant_text(final_state: dict[str, Any]) -> str:
    raw_messages = final_state.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise AgentServerResponseError("Agent Server state has no messages")

    messages = convert_to_messages(
        cast("list[MessageLikeRepresentation]", raw_messages),
    )
    final_message = messages[-1]

    if not isinstance(final_message, AIMessage):
        raise AgentServerResponseError("Agent Server state does not end with an AI message")

    text = final_message.text
    if not text:
        raise AgentServerResponseError("Agent Server returned an empty AI message")

    return text


async def request_discord_reply(
    client: LangGraphClient,
    *,
    assistant_id: str,
    channel_id: int,
    message_id: int,
    content: str,
) -> str:
    if not content:
        raise ValueError("Discord message content must not be empty")

    interaction = InteractionContext(
        source=DISCORD_SOURCE,
        conversation_id=str(channel_id),
    )
    response = cast(
        "object",
        await client.runs.wait(  # pyright: ignore[reportUnknownMemberType]
            discord_thread_id(channel_id),
            assistant_id,
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                        "id": str(message_id),
                    }
                ]
            },
            context=interaction,
            multitask_strategy="enqueue",
            if_not_exists="create",
        ),
    )

    if not isinstance(response, dict):
        raise AgentServerResponseError("Agent Server did not return graph state")

    return _assistant_text(cast("dict[str, Any]", response))


async def send_discord_text(
    channel: discord.DMChannel,
    text: str,
) -> None:
    for start in range(0, len(text), DISCORD_MESSAGE_LIMIT):
        await channel.send(text[start : start + DISCORD_MESSAGE_LIMIT])


class M45DiscordClient(discord.Client):
    def __init__(
        self,
        *,
        agent_server_url: str,
        assistant_id: str,
        allowed_user_id: int,
    ) -> None:
        super().__init__(
            intents=discord.Intents.default(),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self._agent_client = get_client(url=agent_server_url)
        self._assistant_id = assistant_id
        self._allowed_user_id = allowed_user_id

    async def on_ready(self) -> None:
        user_id = self.user.id if self.user is not None else None
        logger.info(
            "Discord client ready (bot_user_id=%s)",
            user_id,
        )

    async def on_message(
        self,
        message: discord.Message,
    ) -> None:
        if message.author.bot:
            return

        if not isinstance(message.channel, discord.DMChannel):
            return

        if message.author.id != self._allowed_user_id:
            logger.warning(
                "Ignored Discord DM from unauthorized user (user_id=%s)",
                message.author.id,
            )
            return

        if message.attachments or message.stickers or not message.content.strip():
            await message.channel.send(TEXT_ONLY_NOTICE)
            return

        try:
            async with message.channel.typing():
                reply = await request_discord_reply(
                    self._agent_client,
                    assistant_id=self._assistant_id,
                    channel_id=message.channel.id,
                    message_id=message.id,
                    content=message.content,
                )
        except Exception:
            logger.exception(
                "Discord message processing failed (channel_id=%s message_id=%s)",
                message.channel.id,
                message.id,
            )
            await message.channel.send(PROCESSING_ERROR_NOTICE)
            return

        await send_discord_text(
            message.channel,
            reply,
        )

    async def close(self) -> None:
        try:
            await self._agent_client.aclose()
        finally:
            await super().close()


def main() -> None:
    settings = load_settings()

    if settings.discord_bot_token is None:
        raise ValueError("DISCORD_BOT_TOKEN is required to run the Discord adapter")

    token = settings.discord_bot_token.get_secret_value()
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN must not be empty")

    allowed_user_id = settings.discord_allowed_user_id
    if allowed_user_id is None:
        raise ValueError("DISCORD_ALLOWED_USER_ID is required to run the Discord adapter")

    logging.basicConfig(
        level=settings.log_level,
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )

    client = M45DiscordClient(
        agent_server_url=str(settings.agent_server_url),
        assistant_id=settings.agent_server_assistant_id,
        allowed_user_id=allowed_user_id,
    )
    client.run(
        token,
        log_handler=None,
    )
