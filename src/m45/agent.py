from itertools import cycle
from typing import Any

from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import InputAgentState, OutputAgentState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import Runnable

type AgentGraph = Runnable[InputAgentState, OutputAgentState[Any]]

SMOKE_RESPONSE = "m45 smoke response"
SMOKE_SYSTEM_PROMPT = "You are m45 running in deterministic smoke mode."


def create_smoke_agent() -> AgentGraph:
    model = GenericFakeChatModel(messages=cycle([SMOKE_RESPONSE]))

    return create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=model,
        tools=[],
        system_prompt=SMOKE_SYSTEM_PROMPT,
        name="m45",
    )


graph = create_smoke_agent()
