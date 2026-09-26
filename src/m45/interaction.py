from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

AGENT_CHAT_UI_SOURCE = "agent_chat_ui"
DISCORD_SOURCE = "discord"


@dataclass(frozen=True, slots=True)
class InteractionContext:
    source: str | None = None
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        if self.source is None and self.conversation_id is None:
            return

        if self.source is None or self.conversation_id is None:
            raise ValueError("interaction source and conversation ID must be provided together")

        if not self.source:
            raise ValueError("interaction source must not be empty")

        if not self.conversation_id:
            raise ValueError("interaction conversation ID must not be empty")


def discord_thread_id(channel_id: int) -> str:
    if channel_id <= 0:
        raise ValueError("Discord channel ID must be greater than zero")

    return str(
        uuid5(
            NAMESPACE_URL,
            f"urn:m45:discord:dm:{channel_id}",
        )
    )
