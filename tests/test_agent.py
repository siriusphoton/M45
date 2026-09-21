import asyncio

import pytest
from langchain.messages import AIMessage, HumanMessage

from m45.agent import (
    create_configured_chat_model,
    create_deterministic_test_agent,
)
from m45.config import Settings


def test_deterministic_agent_runs_asynchronously_without_credentials() -> None:
    agent = create_deterministic_test_agent()

    result = asyncio.run(
        agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="Hello",
                        id="human-message-1",
                    )
                ]
            }
        )
    )

    messages = result["messages"]

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].id == "human-message-1"

    assistant_message = messages[1]

    assert isinstance(assistant_message, AIMessage)
    assert assistant_message.content == "m45 deterministic test response"
    assert assistant_message.id is not None

    second_result = asyncio.run(
        agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="Hello again",
                        id="human-message-2",
                    )
                ]
            }
        )
    )

    second_messages = second_result["messages"]

    assert len(second_messages) == 2
    assert isinstance(second_messages[1], AIMessage)
    assert second_messages[1].content == "m45 deterministic test response"
    assert second_messages[1].id is not None
    assert second_messages[1].id != assistant_message.id


TEST_DATABASE_URL = "postgresql+psycopg://m45:m45@127.0.0.1:5433/m45_test"


def test_google_genai_requires_its_api_key() -> None:
    settings = Settings.model_validate(
        {
            "database_url": TEST_DATABASE_URL,
            "model_provider": "google_genai",
            "google_api_key": None,
        }
    )

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        create_configured_chat_model(settings)


def test_ollama_cloud_requires_its_api_key() -> None:
    settings = Settings.model_validate(
        {
            "database_url": TEST_DATABASE_URL,
            "model_provider": "ollama",
            "ollama_api_key": None,
        }
    )

    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        create_configured_chat_model(settings)
