import asyncio

from langchain.messages import AIMessage, HumanMessage

from m45.agent import create_smoke_agent


def test_smoke_agent_runs_asynchronously_without_credentials() -> None:
    agent = create_smoke_agent()

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
    assert assistant_message.content == "m45 smoke response"
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
    assert second_messages[1].content == "m45 smoke response"
    assert second_messages[1].id is not None
    assert second_messages[1].id != assistant_message.id
