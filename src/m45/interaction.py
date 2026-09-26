from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

AGENT_CHAT_UI_SOURCE = "agent_chat_ui"
DISCORD_SOURCE = "discord"


@dataclass(frozen=True, slots=True)
class InteractionContext:
    source: str
    conversation_id: str

    def __post_init__(self) -> None:
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
