from itertools import cycle
from typing import Any

from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import InputAgentState, OutputAgentState
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from m45.config import Settings

type AgentGraph = Runnable[InputAgentState, OutputAgentState[Any]]

APPLICATION_SYSTEM_PROMPT = "You are m45, a helpful personal assistant."
TEST_RESPONSE = "m45 deterministic test response"
TEST_SYSTEM_PROMPT = "You are m45 running with a deterministic test model."


def _required_secret(
    value: SecretStr | None,
    environment_variable: str,
) -> str:
    if value is None or not (secret := value.get_secret_value()):
        raise ValueError(f"{environment_variable} is required for the selected model provider")
    return secret


def create_configured_chat_model(settings: Settings) -> BaseChatModel:
    if settings.model_provider == "google_genai":
        api_key = _required_secret(
            settings.google_api_key,
            "GOOGLE_API_KEY",
        )

        return init_chat_model(
            settings.model_name_google,
            model_provider="google_genai",
            api_key=api_key,
            vertexai=False,
        )

    api_key = _required_secret(
        settings.ollama_api_key,
        "OLLAMA_API_KEY",
    )

    return init_chat_model(
        settings.model_name_ollama,
        model_provider="ollama",
        base_url=settings.ollama_base_url,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {api_key}",
            }
        },
    )


def create_application_agent(settings: Settings) -> AgentGraph:
    model = create_configured_chat_model(settings)

    return create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=model,
        tools=[],
        system_prompt=APPLICATION_SYSTEM_PROMPT,
        name="m45",
    )


def create_deterministic_test_agent() -> AgentGraph:
    model = GenericFakeChatModel(messages=cycle([TEST_RESPONSE]))

    return create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=model,
        tools=[],
        system_prompt=TEST_SYSTEM_PROMPT,
        name="m45",
    )
