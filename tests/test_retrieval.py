import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from pydantic import Field

from m45.retrieval import (
    RETRIEVAL_MODEL_CALL_LIMIT,
    RETRIEVAL_TOKEN_BUDGET,
    ContextNote,
    EvidenceReference,
    MemoryContext,
    create_memory_retriever,
    retrieve_memory,
)


class ToolCallingModel(GenericFakeChatModel):
    requests: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _handoff(context: MemoryContext) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "MemoryContext", "args": context.model_dump(), "id": "handoff"}],
    )


def test_retriever_carries_history_orientation_and_evidence() -> None:
    queries: list[str] = []

    @tool
    def lookup_memory(query: str) -> dict[str, object]:
        """Find a remembered phone discussion."""
        queries.append(query)
        return {"source_message_id": 701, "content": "User considered a Galaxy S25 FE."}

    expected = MemoryContext(
        notes=[
            ContextNote(
                text="The user considered a Galaxy S25 FE; purchase is not confirmed.",
                references=[EvidenceReference(kind="source_message", id=701)],
            )
        ],
        limitations=[],
    )
    empty = MemoryContext(notes=[], limitations=[])
    model = ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "lookup_memory", "args": {"query": "phone"}, "id": "lookup"}
                    ],
                ),
                _handoff(expected),
                _handoff(empty),
            ]
        )
    )
    retriever = create_memory_retriever(model, [lookup_memory])
    messages: list[AnyMessage] = [
        HumanMessage(content="Can we talk about the phone?", id="user-1"),
        AIMessage(content="What would you like to know?", id="assistant-1"),
        HumanMessage(content="Which one was it?", id="user-2"),
    ]
    result = asyncio.run(
        retrieve_memory(retriever, messages=messages, daily_context="FIRST DAILY SNAPSHOT")
    )
    assert result == expected
    assert queries == ["phone"]
    assert model.requests[0][1:] == messages
    assert len(messages) == 3
    for request in model.requests:
        assert isinstance(request[0], SystemMessage)
        assert request[0].text.count("FIRST DAILY SNAPSHOT") == 1

    second = asyncio.run(
        retrieve_memory(
            retriever,
            messages=[HumanMessage(content="Hello", id="new-user")],
            daily_context="SECOND DAILY SNAPSHOT",
        )
    )
    assert second == empty
    assert queries == ["phone"]
    assert len(model.requests[-1]) == 2
    assert "FIRST DAILY SNAPSHOT" not in model.requests[-1][0].text


def test_retriever_preserves_tool_failure_even_when_model_omits_it() -> None:
    @tool
    def lookup_memory(query: str) -> str:
        """Find original memory evidence."""
        raise ConnectionError("simulated embedding outage")

    model = ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "lookup_memory", "args": {"query": "phone"}, "id": "lookup"}
                    ],
                ),
                _handoff(MemoryContext(notes=[], limitations=[])),
            ]
        )
    )
    result = asyncio.run(
        retrieve_memory(
            create_memory_retriever(model, [lookup_memory]),
            messages=[HumanMessage(content="Which phone?")],
            daily_context="No daily log available.",
        )
    )
    assert result.notes == []
    assert any("lookup_memory" in limitation for limitation in result.limitations)
    assert "Unavailable" in result.limitations[-1]


def test_retriever_stops_oversized_input_and_exhausted_runs() -> None:
    model = ToolCallingModel(messages=iter([]))
    with pytest.raises(ValueError, match="retrieval input exceeds budget"):
        asyncio.run(
            retrieve_memory(
                create_memory_retriever(model, []),
                messages=[HumanMessage(content="Hello")],
                daily_context="x" * (RETRIEVAL_TOKEN_BUDGET * 4),
            )
        )
    assert model.requests == []

    model = ToolCallingModel(
        messages=iter(
            AIMessage(
                content="",
                tool_calls=[{"name": "MemoryContext", "args": {}, "id": f"bad-{index}"}],
            )
            for index in range(RETRIEVAL_MODEL_CALL_LIMIT + 1)
        )
    )
    result = asyncio.run(
        retrieve_memory(
            create_memory_retriever(model, []),
            messages=[HumanMessage(content="Hello")],
            daily_context="No daily log available.",
        )
    )
    assert len(model.requests) == RETRIEVAL_MODEL_CALL_LIMIT
    assert result.notes == [] and result.limitations
    assert "Model call" in result.limitations[0]

    reads: list[str] = []

    @tool
    def read_memory(query: str) -> str:
        """Read original evidence."""
        reads.append(query)
        return "x" * (RETRIEVAL_TOKEN_BUDGET * 4) if query == "oversized" else "Evidence"

    model = ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_memory", "args": {"query": "oversized"}, "id": "large"}
                    ],
                )
            ]
        )
    )
    with pytest.raises(ValueError, match="retrieval input exceeds budget"):
        asyncio.run(
            retrieve_memory(
                create_memory_retriever(model, [read_memory]),
                messages=[HumanMessage(content="Which phone?")],
                daily_context="No daily log available.",
            )
        )
    assert len(model.requests) == 1
    reads.clear()

    model = ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_memory", "args": {"query": "phone"}, "id": str(index)}
                        for index in range(11)
                    ],
                )
            ]
        )
    )
    result = asyncio.run(
        retrieve_memory(
            create_memory_retriever(model, [read_memory]),
            messages=[HumanMessage(content="Which phone?")],
            daily_context="No daily log available.",
        )
    )
    assert reads == []
    assert result.notes == [] and result.limitations
    assert "Tool call" in result.limitations[0]


def test_incomplete_extra_retrieval_preserves_previous_evidence() -> None:
    previous = MemoryContext(
        notes=[
            ContextNote(
                text="The user considered a Galaxy S25 FE; purchase is unconfirmed.",
                references=[EvidenceReference(kind="source_message", id=701)],
            )
        ],
        limitations=["Purchase is unconfirmed."],
    )
    model = ToolCallingModel(
        messages=iter(
            AIMessage(
                content="",
                tool_calls=[{"name": "MemoryContext", "args": {}, "id": f"bad-{index}"}],
            )
            for index in range(RETRIEVAL_MODEL_CALL_LIMIT + 1)
        )
    )
    result = asyncio.run(
        retrieve_memory(
            create_memory_retriever(model, []),
            messages=[HumanMessage(content="Did I buy it?")],
            daily_context="No daily log available.",
            focus="Find a purchase confirmation.",
            previous_context=previous,
        )
    )
    assert result.notes == previous.notes
    assert result.limitations[0] == "Purchase is unconfirmed."
    assert any("Model call" in limitation for limitation in result.limitations)
    assert "Find a purchase confirmation." in model.requests[0][0].text
    assert "Galaxy S25 FE" in model.requests[0][0].text
