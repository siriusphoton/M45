import asyncio
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    OutputAgentState,
    ToolCallLimitMiddleware,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, StateSnapshot
from pydantic import SecretStr
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45.agent import (
    APPLICATION_SYSTEM_PROMPT,
    FIXED_SYSTEM_PROMPT,
    create_application_agent,
)
from m45.agent_middleware import (
    MAIN_TOKEN_BUDGET,
    REQUEST_CONTEXT_TOOL_NAME,
    CurrentTimeMiddleware,
    InteractionInstructionsMiddleware,
    MemoryContextMiddleware,
    PersonalContextMiddleware,
    SourceHistoryMiddleware,
    StripResponseTimestampMiddleware,
)
from m45.config import Settings, load_settings
from m45.interaction import InteractionContext
from m45.retrieval import (
    ContextNote,
    EvidenceReference,
    MemoryContext,
    create_memory_retriever,
)
from m45.source_history import SourceMessage, capture_source_message
from m45.web_search import WEB_SEARCH_TOOL_NAME

TEST_SOURCE = "agent_middleware_test"
OTHER_CONVERSATION_ID = "middleware-context-other"
CURRENT_CONVERSATION_ID = "middleware-context-current"
TOOL_CONVERSATION_ID = "middleware-tool-current"
TIME_CONVERSATION_ID = "middleware-time-thread"
TEST_CONVERSATION_IDS = (
    OTHER_CONVERSATION_ID,
    CURRENT_CONVERSATION_ID,
    TOOL_CONVERSATION_ID,
    TIME_CONVERSATION_ID,
)
EXPLICIT_CONTEXT_THREAD_ID = "middleware-context-checkpoint-thread"
FIXED_CURRENT_TIME = datetime(
    2026,
    9,
    27,
    21,
    15,
    tzinfo=ZoneInfo("Asia/Kolkata"),
)


class ModelInputRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.model_inputs: list[list[BaseMessage]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_inputs.extend(messages)


class ToolCallingFakeChatModel(GenericFakeChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


@pytest.fixture
def source_history_test_data(engine: Engine) -> Iterator[None]:
    with Session(engine) as session, session.begin():
        session.execute(
            delete(SourceMessage).where(
                SourceMessage.source == TEST_SOURCE,
                SourceMessage.conversation_id.in_(
                    TEST_CONVERSATION_IDS,
                ),
            )
        )
        previous_message = capture_source_message(
            session,
            source=TEST_SOURCE,
            conversation_id=OTHER_CONVERSATION_ID,
            message_id="prior-user-message",
            role="user",
            content="My preferred editor is Neovim.",
        )
        previous_message.captured_at = datetime(2026, 9, 27, 8, 35, tzinfo=UTC)
        earlier_same_conversation = capture_source_message(
            session,
            source=TEST_SOURCE,
            conversation_id=CURRENT_CONVERSATION_ID,
            message_id="before-checkpoint-reset",
            role="user",
            content="I moved to Neovim last month.",
        )
        earlier_same_conversation.captured_at = datetime(2026, 9, 27, 8, 36, tzinfo=UTC)

    try:
        yield
    finally:
        with Session(engine) as session, session.begin():
            session.execute(
                delete(SourceMessage).where(
                    SourceMessage.source == TEST_SOURCE,
                    SourceMessage.conversation_id.in_(
                        TEST_CONVERSATION_IDS,
                    ),
                )
            )


@pytest.mark.parametrize(
    ("thread_id", "interaction_context"),
    [
        pytest.param(
            CURRENT_CONVERSATION_ID,
            InteractionContext(),
            id="agent-chat-ui-empty-context",
        ),
        pytest.param(
            EXPLICIT_CONTEXT_THREAD_ID,
            InteractionContext(
                source=TEST_SOURCE,
                conversation_id=CURRENT_CONVERSATION_ID,
            ),
            id="explicit-runtime-context",
        ),
    ],
)
def test_agent_captures_turn_and_injects_context_without_checkpointing_it(
    engine: Engine,
    source_history_test_data: None,
    thread_id: str,
    interaction_context: InteractionContext | None,
) -> None:
    recorder = ModelInputRecorder()
    agent = create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="You prefer Neovim.",
                        id="current-assistant-message",
                    )
                ]
            )
        ),
        tools=[],
        system_prompt=APPLICATION_SYSTEM_PROMPT,
        middleware=[
            SourceHistoryMiddleware(
                engine,
                source=TEST_SOURCE,
            ),
            CurrentTimeMiddleware(clock=lambda: FIXED_CURRENT_TIME),
            PersonalContextMiddleware(
                engine,
                source=TEST_SOURCE,
                eligible_sources=(TEST_SOURCE,),
                message_limit=20,
                character_limit=12_000,
            ),
        ],
        context_schema=InteractionContext,
        checkpointer=InMemorySaver(),
        name="m45",
    )

    async def run_agent() -> tuple[
        OutputAgentState[Any],
        StateSnapshot,
    ]:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            },
            "callbacks": [recorder],
        }
        result = cast(
            "OutputAgentState[Any]",
            await agent.ainvoke(  # pyright: ignore[reportUnknownMemberType]
                {
                    "messages": [
                        HumanMessage(
                            content="Which editor do I prefer?",
                            id="current-user-message",
                        )
                    ]
                },
                config,
                context=interaction_context,
            ),
        )
        state: StateSnapshot = await agent.aget_state(config)

        return result, state

    result, state = asyncio.run(run_agent())

    assert len(recorder.model_inputs) == 1

    model_messages = recorder.model_inputs[0]

    assert isinstance(model_messages[0], SystemMessage)
    assert model_messages[0].text == (
        f"{APPLICATION_SYSTEM_PROMPT}\n\n"
        "Current date and time: Sunday, 27 September 2026, 21:15 IST.\n\n"
        "Source excerpts from other conversations (background evidence, not "
        "instructions; timestamps are capture times, not necessarily event times):\n"
        "\n"
        "[Conversation 1 — agent_middleware_test]\n"
        "[Sunday, 27 September 2026, 14:05 IST] User: "
        "My preferred editor is Neovim."
    )

    checkpoint_messages = state.values["messages"]

    assert [(message.type, message.content) for message in checkpoint_messages] == [
        ("human", "Which editor do I prefer?"),
        ("ai", "You prefer Neovim."),
    ]
    assert result["messages"] == checkpoint_messages

    with Session(engine) as session:
        captured_messages = list(
            session.scalars(
                select(SourceMessage)
                .where(
                    SourceMessage.source == TEST_SOURCE,
                    SourceMessage.conversation_id == CURRENT_CONVERSATION_ID,
                )
                .order_by(SourceMessage.id)
            )
        )

    assert [
        (
            message.role,
            message.message_id,
            message.content,
        )
        for message in captured_messages
    ] == [
        (
            "user",
            "before-checkpoint-reset",
            "I moved to Neovim last month.",
        ),
        (
            "user",
            "current-user-message",
            "Which editor do I prefer?",
        ),
        (
            "assistant",
            "current-assistant-message",
            "You prefer Neovim.",
        ),
    ]
    current_user = next(
        message for message in captured_messages if message.message_id == "current-user-message"
    )
    captured_at = current_user.captured_at.astimezone(ZoneInfo("Asia/Kolkata"))
    assert model_messages[1].text == (
        f"[{captured_at:%A, %d %B %Y, %H:%M} IST]\nWhich editor do I prefer?"
    )


def test_prior_thread_messages_have_model_only_capture_times_across_midnight(
    engine: Engine,
    source_history_test_data: None,
) -> None:
    recorder = ModelInputRecorder()
    agent = create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=GenericFakeChatModel(  # type: ignore
            messages=iter(
                [
                    AIMessage(content="First answer", id="late-assistant"),
                    AIMessage(content="Second answer", id="today-assistant"),
                ]
            )
        ),
        tools=[],
        system_prompt="Temporal test",
        middleware=[SourceHistoryMiddleware(engine, source=TEST_SOURCE)],
        context_schema=InteractionContext,
        checkpointer=InMemorySaver(),
        name="m45",
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": TIME_CONVERSATION_ID},
        "callbacks": [recorder],
    }
    context = InteractionContext(
        source=TEST_SOURCE,
        conversation_id=TIME_CONVERSATION_ID,
    )

    agent.invoke(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="Late greeting", id="late-user")]},
        config,
        context=context,
    )
    with Session(engine) as session, session.begin():
        prior_messages = session.scalars(
            select(SourceMessage).where(
                SourceMessage.source == TEST_SOURCE,
                SourceMessage.conversation_id == TIME_CONVERSATION_ID,
            )
        ).all()
        capture_times = {
            "late-user": datetime(2026, 9, 28, 18, 27, tzinfo=UTC),
            "late-assistant": datetime(2026, 9, 28, 18, 28, tzinfo=UTC),
        }
        for message in prior_messages:
            message.captured_at = capture_times[message.message_id]

    agent.invoke(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="Was that yesterday?", id="today-user")]},
        config,
        context=context,
    )

    model_messages = recorder.model_inputs[1]
    assert model_messages[1].text == ("[Monday, 28 September 2026, 23:57 IST]\nLate greeting")
    assert model_messages[2].text == ("[Monday, 28 September 2026, 23:58 IST]\nFirst answer")
    assert model_messages[3].text.endswith("\nWas that yesterday?")

    state = agent.get_state(config)  # pyright: ignore[reportUnknownMemberType]
    assert [(message.type, message.content) for message in state.values["messages"]] == [
        ("human", "Late greeting"),
        ("ai", "First answer"),
        ("human", "Was that yesterday?"),
        ("ai", "Second answer"),
    ]


def test_tool_turn_checkpoints_evidence_and_captures_only_final_messages(
    engine: Engine,
    source_history_test_data: None,
) -> None:
    @tool(WEB_SEARCH_TOOL_NAME)
    def web_search(query: str) -> str:
        """Search the public web."""
        return (
            "Title: PostgreSQL release notes\n"
            "URL: https://www.postgresql.org/docs/release/\n"
            f"Snippet: Results for {query}."
        )

    agent = create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=ToolCallingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        id="search-request-message",
                        tool_calls=[
                            {
                                "name": WEB_SEARCH_TOOL_NAME,
                                "args": {
                                    "query": "current PostgreSQL release",
                                },
                                "id": "search-call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content=(
                            "The release notes are available at "
                            "https://www.postgresql.org/docs/release/."
                        ),
                        id="final-assistant-message",
                    ),
                ]
            )
        ),
        tools=[web_search],
        middleware=[
            SourceHistoryMiddleware(
                engine,
                source=TEST_SOURCE,
            ),
        ],
        context_schema=InteractionContext,
        checkpointer=InMemorySaver(),
        name="m45",
    )
    config: RunnableConfig = {
        "configurable": {
            "thread_id": TOOL_CONVERSATION_ID,
        }
    }

    async def run_agent() -> tuple[
        OutputAgentState[Any],
        StateSnapshot,
    ]:
        result = cast(
            "OutputAgentState[Any]",
            await agent.ainvoke(  # pyright: ignore[reportUnknownMemberType]
                {
                    "messages": [
                        HumanMessage(
                            content="What is the current PostgreSQL release?",
                            id="tool-turn-user-message",
                        )
                    ]
                },
                config,
                context=InteractionContext(
                    source=TEST_SOURCE,
                    conversation_id=TOOL_CONVERSATION_ID,
                ),
            ),
        )
        state: StateSnapshot = await agent.aget_state(config)

        return result, state

    result, state = asyncio.run(run_agent())
    checkpoint_messages = state.values["messages"]

    assert [message.type for message in checkpoint_messages] == [
        "human",
        "ai",
        "tool",
        "ai",
    ]
    assert isinstance(checkpoint_messages[2], ToolMessage)
    assert "https://www.postgresql.org/docs/release/" in checkpoint_messages[2].text
    assert result["messages"] == checkpoint_messages

    with Session(engine) as session:
        captured_messages = list(
            session.scalars(
                select(SourceMessage)
                .where(
                    SourceMessage.source == TEST_SOURCE,
                    SourceMessage.conversation_id == TOOL_CONVERSATION_ID,
                )
                .order_by(SourceMessage.id)
            )
        )

    assert [
        (
            message.role,
            message.message_id,
            message.content,
        )
        for message in captured_messages
    ] == [
        (
            "user",
            "tool-turn-user-message",
            "What is the current PostgreSQL release?",
        ),
        (
            "assistant",
            "final-assistant-message",
            ("The release notes are available at https://www.postgresql.org/docs/release/."),
        ),
    ]


def test_strip_response_timestamp_middleware(
    engine: Engine,
    source_history_test_data: None,
) -> None:
    agent = create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="[Tuesday, 29 September 2026, 23:10 IST]\nFair enough.",
                        id="tsrtm-1",
                    ),
                ]
            )
        ),
        tools=[],
        system_prompt="Temporal test",
        middleware=[
            SourceHistoryMiddleware(
                engine,
                source=TEST_SOURCE,
            ),
            StripResponseTimestampMiddleware(),
        ],
        context_schema=InteractionContext,
        checkpointer=InMemorySaver(),
        name="m45",
    )
    config: RunnableConfig = {
        "configurable": {
            "thread_id": CURRENT_CONVERSATION_ID,
        },
    }
    agent.invoke(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="Hi", id="m1")]}, config=config
    )

    state: StateSnapshot = agent.get_state(config=config)
    checkpoint_messages = state.values["messages"]

    assert checkpoint_messages[-1].content == "Fair enough."
    assert checkpoint_messages[-1].id == "tsrtm-1"

    with Session(engine) as session:
        captured_reply = session.scalars(
            select(SourceMessage).where(
                SourceMessage.source == TEST_SOURCE,
                SourceMessage.conversation_id == CURRENT_CONVERSATION_ID,
                SourceMessage.message_id == "tsrtm-1",
            )
        ).one()

    assert captured_reply.content == "Fair enough."


def test_memory_context_survives_review_and_is_replaced_on_new_input(
    engine: Engine,
    source_history_test_data: None,
) -> None:
    recorder = ModelInputRecorder()
    first_context = MemoryContext(notes=[], limitations=["FIRST RETRIEVAL GAP"])
    first_focused = MemoryContext(notes=[], limitations=["FIRST FOCUSED GAP"])
    second_context = MemoryContext(notes=[], limitations=["SECOND RETRIEVAL GAP"])
    second_focused = MemoryContext(notes=[], limitations=["SECOND FOCUSED GAP"])
    retriever = create_memory_retriever(
        ToolCallingFakeChatModel(
            messages=iter(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "MemoryContext",
                            "args": context.model_dump(),
                            "id": f"format-{index}",
                        }
                    ],
                )
                for index, context in enumerate(
                    (first_context, first_focused, second_context, second_focused)
                )
            )
        ),
        [],
    )

    @tool
    def review_change() -> str:
        """Perform the reviewed test action."""
        return "Approved test action completed."

    preparation_times = iter([FIXED_CURRENT_TIME, FIXED_CURRENT_TIME + timedelta(days=1)])
    middleware: list[AgentMiddleware[Any, InteractionContext, Any]] = [
        SourceHistoryMiddleware(engine, source=TEST_SOURCE),
        MemoryContextMiddleware(
            engine,
            retriever,
            source=TEST_SOURCE,
            eligible_sources=(TEST_SOURCE,),
            clock=lambda: next(preparation_times),
        ),
        ToolCallLimitMiddleware[Any, InteractionContext](
            tool_name=REQUEST_CONTEXT_TOOL_NAME, run_limit=1, exit_behavior="continue"
        ),
        HumanInTheLoopMiddleware(interrupt_on={"review_change": True}),
    ]
    saver = InMemorySaver()
    agent = create_agent(  # pyright: ignore[reportUnknownVariableType]
        model=ToolCallingFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="Investigating a gap.",
                        id="first-context-request",
                        tool_calls=[
                            {
                                "name": REQUEST_CONTEXT_TOOL_NAME,
                                "args": {"focus": "FIRST SPECIFIC GAP"},
                                "id": "first-focus",
                            },
                            {
                                "name": REQUEST_CONTEXT_TOOL_NAME,
                                "args": {"focus": "DUPLICATE GAP"},
                                "id": "duplicate-focus",
                            },
                        ],
                    ),
                    AIMessage(
                        content="",
                        id="handoff-review-message",
                        tool_calls=[{"name": "review_change", "args": {}, "id": "review"}],
                    ),
                    AIMessage(
                        content="",
                        id="repeated-context-request",
                        tool_calls=[
                            {
                                "name": REQUEST_CONTEXT_TOOL_NAME,
                                "args": {"focus": "REPEATED GAP"},
                                "id": "repeated-focus",
                            }
                        ],
                    ),
                    AIMessage(content="First turn completed.", id="handoff-answer-1"),
                    AIMessage(
                        content="",
                        id="second-context-request",
                        tool_calls=[
                            {
                                "name": REQUEST_CONTEXT_TOOL_NAME,
                                "args": {"focus": "SECOND SPECIFIC GAP"},
                                "id": "second-focus",
                            }
                        ],
                    ),
                    AIMessage(content="Second turn completed.", id="handoff-answer-2"),
                ]
            )
        ),
        tools=[review_change],
        system_prompt="Handoff lifecycle test",
        middleware=middleware,
        context_schema=InteractionContext,
        checkpointer=saver,
    )

    async def run_agent() -> tuple[StateSnapshot, str]:
        config: RunnableConfig = {
            "configurable": {"thread_id": TOOL_CONVERSATION_ID},
            "callbacks": [recorder],
            "run_id": uuid4(),
        }
        interaction = InteractionContext(source=TEST_SOURCE, conversation_id=TOOL_CONVERSATION_ID)
        interrupted = await agent.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            {"messages": [HumanMessage(content="First question", id="handoff-user-1")]},
            config,
            context=interaction,
        )
        assert "__interrupt__" in interrupted
        paused: StateSnapshot = await agent.aget_state(config)
        first_daily_context = str(paused.values["daily_context"])
        assert paused.values["memory_context"] == first_focused.model_dump_json(indent=2)
        assert paused.values["extra_context_used"] is True

        resumed = await agent.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            Command(resume={"decisions": [{"type": "approve"}]}),
            {**config, "run_id": uuid4()},
            context=interaction,
        )
        after_review: StateSnapshot = await agent.aget_state(config)
        assert after_review.values["daily_context"] == first_daily_context
        assert after_review.values["memory_context"] == paused.values["memory_context"]
        assert "daily_context" not in resumed and "memory_context" not in resumed

        result = await agent.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            {"messages": [HumanMessage(content="Second question", id="handoff-user-2")]},
            {**config, "run_id": uuid4()},
            context=interaction,
        )
        assert "daily_context" not in result and "memory_context" not in result
        final_state: StateSnapshot = await agent.aget_state(config)
        return final_state, first_daily_context

    final_state, first_daily_context = asyncio.run(run_agent())
    assert final_state.values["daily_context"] != first_daily_context
    assert final_state.values["memory_context"] == second_focused.model_dump_json(indent=2)

    model_inputs = [
        messages
        for messages in recorder.model_inputs
        if messages[0].text.startswith("Handoff lifecycle test")
    ]
    assert len(model_inputs) == 6
    for messages, marker in zip(
        model_inputs,
        (
            "FIRST RETRIEVAL GAP",
            "FIRST FOCUSED GAP",
            "FIRST FOCUSED GAP",
            "FIRST FOCUSED GAP",
            "SECOND RETRIEVAL GAP",
            "SECOND FOCUSED GAP",
        ),
        strict=True,
    ):
        assert messages[0].text.count(marker) == 1
        assert messages[0].text.count("Daily context: background evidence") == 1
    assert "FIRST FOCUSED GAP" not in model_inputs[-1][0].text

    retrieval_inputs = [
        messages
        for messages in recorder.model_inputs
        if messages[0].text.startswith("You retrieve personal context")
    ]
    assert len(retrieval_inputs) == 4
    assert [message.id for message in retrieval_inputs[1][1:]] == ["handoff-user-1"]
    assert "FIRST SPECIFIC GAP" in retrieval_inputs[1][0].text
    assert "FIRST RETRIEVAL GAP" in retrieval_inputs[1][0].text
    assert "SECOND SPECIFIC GAP" in retrieval_inputs[3][0].text
    assert "SECOND RETRIEVAL GAP" in retrieval_inputs[3][0].text
    previous_ids = {message.id for message in retrieval_inputs[3][1:]}
    assert {"handoff-user-1", "handoff-answer-1", "handoff-user-2"} <= previous_ids

    assert "second-context-request" not in previous_ids
    first_human = next(
        message.text for message in model_inputs[0] if message.id == "handoff-user-1"
    )
    assert first_human.endswith("\nFirst question")
    assert retrieval_inputs[0][1].text == first_human
    assert retrieval_inputs[1][1].text == first_human
    for message_id in ("handoff-answer-1", "handoff-user-2"):
        expected = next(message.text for message in model_inputs[4] if message.id == message_id)
        for messages in retrieval_inputs[2:]:
            actual = next(message.text for message in messages if message.id == message_id)
            assert actual == expected

    raw_messages = {
        "handoff-user-1": "First question",
        "handoff-answer-1": "First turn completed.",
        "handoff-user-2": "Second question",
        "handoff-answer-2": "Second turn completed.",
    }
    for checkpoint in saver.list({"configurable": {"thread_id": TOOL_CONVERSATION_ID}}):
        for message in checkpoint.checkpoint["channel_values"].get("messages", []):
            if message.id in raw_messages:
                assert message.content == raw_messages[message.id]
    tool_messages = [
        message for message in final_state.values["messages"] if isinstance(message, ToolMessage)
    ]
    statuses = {message.tool_call_id: message.status for message in tool_messages}
    assert statuses["duplicate-focus"] == "error"
    assert statuses["repeated-focus"] == "error"
    assert statuses["first-focus"] == statuses["second-focus"] == "success"
    assert all(message.name != "MemoryContext" for message in tool_messages)
    assert all("FOCUSED GAP" not in message.text for message in tool_messages)

    with Session(engine) as session:
        rows = session.execute(
            select(SourceMessage.message_id, SourceMessage.content).where(
                SourceMessage.source == TEST_SOURCE,
                SourceMessage.conversation_id == TOOL_CONVERSATION_ID,
            )
        )
        captured_messages = {message_id: content for message_id, content in rows}
    assert captured_messages == raw_messages


def test_application_agent_retrieves_context_and_budgets_injected_input(
    engine: Engine,
    source_history_test_data: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(engine) as session:
        source_id = session.scalars(
            select(SourceMessage.id).where(
                SourceMessage.source == TEST_SOURCE,
                SourceMessage.conversation_id == OTHER_CONVERSATION_ID,
                SourceMessage.message_id == "prior-user-message",
            )
        ).one()
    reference = EvidenceReference(kind="source_message", id=source_id)
    initial = MemoryContext(
        notes=[ContextNote(text="The user preferred Neovim.", references=[reference])],
        limitations=[],
    )
    focused = MemoryContext(
        notes=[
            ContextNote(
                text="Neovim was reported; no later change is known.", references=[reference]
            )
        ],
        limitations=[],
    )

    def handoff(context: MemoryContext, call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{"name": "MemoryContext", "args": context.model_dump(), "id": call_id}],
        )

    model = ToolCallingFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_source_message",
                            "args": {"source_message_id": source_id},
                            "id": "factory-read",
                        }
                    ],
                ),
                handoff(initial, "factory-initial"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": REQUEST_CONTEXT_TOOL_NAME,
                            "args": {"focus": "Has that preference changed?"},
                            "id": f"factory-focus-{index}",
                        }
                        for index in range(2)
                    ],
                ),
                handoff(focused, "factory-focused"),
                AIMessage(content="Neovim.", id="factory-answer"),
                handoff(initial, "factory-next"),
            ]
        )
    )

    def configured_test_model(_settings: Settings) -> ToolCallingFakeChatModel:
        return model

    monkeypatch.setattr("m45.agent.create_configured_chat_model", configured_test_model)
    settings = load_settings().model_copy(update={"ollama_api_key": SecretStr("unused-test-key")})
    agent = create_application_agent(settings, engine)
    recorder = ModelInputRecorder()
    config: RunnableConfig = {
        "configurable": {"thread_id": TOOL_CONVERSATION_ID},
        "callbacks": [recorder],
    }
    interaction = InteractionContext(source=TEST_SOURCE, conversation_id=TOOL_CONVERSATION_ID)
    result = asyncio.run(
        agent.ainvoke(
            {"messages": [HumanMessage(content="Which editor do I prefer?", id="factory-user-1")]},
            config,
            context=interaction,
        )
    )
    assert result["messages"][-1].text == "Neovim."
    assert "daily_context" not in result and "memory_context" not in result
    assert len(recorder.model_inputs) == 5
    main_requests = [
        messages
        for messages in recorder.model_inputs
        if messages[0].text.startswith(FIXED_SYSTEM_PROMPT)
    ]
    assert initial.notes[0].text in main_requests[0][0].text
    assert focused.notes[0].text in main_requests[1][0].text
    assert "Standing interaction guidance" in main_requests[1][0].text
    assert "Current date and time" in main_requests[1][0].text
    assert any(
        isinstance(message, ToolMessage)
        and message.name == "read_source_message"
        and "My preferred editor is Neovim." in message.text
        for messages in recorder.model_inputs
        for message in messages
    )
    statuses = {
        message.tool_call_id: message.status
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }
    assert statuses["factory-focus-0"] == "success"
    assert statuses["factory-focus-1"] == "error"
    with Session(engine) as session:
        rows = session.scalars(
            select(SourceMessage)
            .where(
                SourceMessage.source == TEST_SOURCE,
                SourceMessage.conversation_id == TOOL_CONVERSATION_ID,
            )
            .order_by(SourceMessage.id)
        ).all()
    assert [(row.role, row.content) for row in rows] == [
        ("user", "Which editor do I prefer?"),
        ("assistant", "Neovim."),
    ]

    def oversized_guidance(_middleware: InteractionInstructionsMiddleware) -> str:
        return "x" * (MAIN_TOKEN_BUDGET * 4)

    monkeypatch.setattr(InteractionInstructionsMiddleware, "_load", oversized_guidance)
    with pytest.raises(ValueError, match="main model input exceeds budget"):
        asyncio.run(
            agent.ainvoke(
                {"messages": [HumanMessage(content="Another question", id="factory-user-2")]},
                config,
                context=interaction,
            )
        )
    assert len(recorder.model_inputs) == 6
