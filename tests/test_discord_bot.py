import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import discord
from langgraph_sdk.client import LangGraphClient

from m45.discord_bot import (
    M45DiscordClient,
    request_discord_reply,
    send_discord_text,
)
from m45.interaction import (
    DISCORD_SOURCE,
    InteractionContext,
    discord_thread_id,
)


def test_request_discord_reply_preserves_source_identity() -> None:
    wait = AsyncMock(
        return_value={
            "messages": [
                {
                    "type": "human",
                    "content": "Hello",
                    "id": "456",
                },
                {
                    "type": "ai",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello back",
                        }
                    ],
                    "id": "assistant-1",
                },
            ]
        }
    )
    client = cast(
        "LangGraphClient",
        SimpleNamespace(
            runs=SimpleNamespace(wait=wait),
        ),
    )

    reply = asyncio.run(
        request_discord_reply(
            client,
            assistant_id="m45",
            channel_id=123,
            message_id=456,
            content="Hello",
        )
    )

    assert reply == "Hello back"
    wait.assert_awaited_once_with(
        discord_thread_id(123),
        "m45",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "id": "456",
                }
            ]
        },
        context=InteractionContext(
            source=DISCORD_SOURCE,
            conversation_id="123",
        ),
        multitask_strategy="enqueue",
        if_not_exists="create",
    )


def test_send_discord_text_splits_replies_at_platform_limit() -> None:
    send = AsyncMock()
    channel = cast(
        "discord.DMChannel",
        SimpleNamespace(send=send),
    )

    asyncio.run(
        send_discord_text(
            channel,
            "a" * 2_001,
        )
    )

    send.assert_has_awaits(
        [
            call("a" * 2_000),
            call("a"),
        ]
    )


def test_discord_client_ignores_dm_from_unauthorized_user() -> None:
    raw_channel = MagicMock(spec=discord.DMChannel)
    raw_channel.send = AsyncMock()
    channel = cast("discord.DMChannel", raw_channel)

    raw_message = MagicMock(spec=discord.Message)
    raw_message.author = SimpleNamespace(
        id=999,
        bot=False,
    )
    raw_message.channel = channel
    message = cast("discord.Message", raw_message)

    client = M45DiscordClient(
        agent_server_url="http://127.0.0.1:2024",
        assistant_id="m45",
        allowed_user_id=123,
    )

    with patch(
        "m45.discord_bot.request_discord_reply",
        new_callable=AsyncMock,
    ) as request_reply:
        asyncio.run(client.on_message(message))

    request_reply.assert_not_awaited()
    raw_channel.send.assert_not_awaited()
    asyncio.run(client.close())
