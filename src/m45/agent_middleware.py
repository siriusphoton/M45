import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from m45.interaction import AGENT_CHAT_UI_SOURCE, InteractionContext
from m45.personal_context import (
    format_recent_personal_context,
    load_recent_personal_context,
)
from m45.source_history import MessageRole, capture_source_message

INDIA_TIME_ZONE = ZoneInfo("Asia/Kolkata")


def _current_time_in_india() -> datetime:
    return datetime.now(INDIA_TIME_ZONE)


def _thread_id(runtime: Runtime[Any]) -> str:
    execution_info = runtime.execution_info

    if execution_info is None or execution_info.thread_id is None:
        raise ValueError("a LangGraph thread_id is required")

    return execution_info.thread_id


def _append_system_text(
    request: ModelRequest[InteractionContext],
    text: str,
) -> ModelRequest[InteractionContext]:
    content_blocks = (
        list(request.system_message.content_blocks) if request.system_message is not None else []
    )
    block_text = f"\n\n{text}" if content_blocks else text
    content_blocks.append({"type": "text", "text": block_text})

    return request.override(
        system_message=SystemMessage(
            content=cast("list[str | dict[str, Any]]", content_blocks),
        ),
    )


def _interaction_identity(
    runtime: Runtime[InteractionContext],
    *,
    default_source: str,
) -> tuple[str, str]:
    context = cast(InteractionContext | None, runtime.context)

    if context is None or context.source is None or context.conversation_id is None:
        return default_source, _thread_id(runtime)

    return context.source, context.conversation_id


def _text_content(
    message: BaseMessage,
    *,
    role: MessageRole,
) -> str:
    if role == "assistant":
        text = message.text
    else:
        blocks = message.content_blocks

        if not blocks:
            raise ValueError("message content must not be empty")

        text_parts: list[str] = []

        for block in blocks:
            if block["type"] != "text":
                raise ValueError("m45 currently supports text-only user messages")

            text_parts.append(block["text"])

        text = "".join(text_parts)

    if not text:
        raise ValueError("message content must not be empty")

    return text


def _message_id(message: BaseMessage) -> str:
    if message.id is None or not message.id:
        raise ValueError("source messages require a message ID")

    return message.id


class SourceHistoryMiddleware(
    AgentMiddleware[AgentState[Any], InteractionContext, Any],
):
    tools = ()

    def __init__(
        self,
        bind: Engine | Connection,
        *,
        source: str = AGENT_CHAT_UI_SOURCE,
    ) -> None:
        self._bind = bind
        self._source = source

    def _capture(
        self,
        state: AgentState[Any],
        runtime: Runtime[InteractionContext],
        *,
        role: MessageRole,
    ) -> None:
        messages = state["messages"]
        expected_type = HumanMessage if role == "user" else AIMessage

        source, conversation_id = _interaction_identity(
            runtime,
            default_source=self._source,
        )

        if not messages or not isinstance(
            message := messages[-1],
            expected_type,
        ):
            raise ValueError(f"agent state must end with a {role} message")

        with Session(self._bind) as session, session.begin():
            capture_source_message(
                session,
                source=source,
                conversation_id=conversation_id,
                message_id=_message_id(message),
                role=role,
                content=_text_content(message, role=role),
            )

    def before_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[InteractionContext],
    ) -> None:
        self._capture(state, runtime, role="user")

    async def abefore_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[InteractionContext],
    ) -> None:
        await asyncio.to_thread(
            self._capture,
            state,
            runtime,
            role="user",
        )

    def after_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[InteractionContext],
    ) -> None:
        self._capture(state, runtime, role="assistant")

    async def aafter_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[InteractionContext],
    ) -> None:
        await asyncio.to_thread(
            self._capture,
            state,
            runtime,
            role="assistant",
        )


class CurrentTimeMiddleware(
    AgentMiddleware[AgentState[Any], InteractionContext, Any],
):
    tools = ()

    def __init__(
        self,
        clock: Callable[[], datetime] = _current_time_in_india,
    ) -> None:
        self._clock = clock

    def _context(self) -> str:
        current_time = self._clock().astimezone(INDIA_TIME_ZONE)
        return (
            f"Current date and time in India: {current_time:%A, %d %B %Y at %H:%M} IST (UTC+05:30)."
        )

    def wrap_model_call(
        self,
        request: ModelRequest[InteractionContext],
        handler: Callable[
            [ModelRequest[InteractionContext]],
            ModelResponse[Any],
        ],
    ) -> ModelCallResult[Any]:
        return handler(
            _append_system_text(request, self._context()),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[InteractionContext],
        handler: Callable[
            [ModelRequest[InteractionContext]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelCallResult[Any]:
        return await handler(
            _append_system_text(request, self._context()),
        )


class PersonalContextMiddleware(
    AgentMiddleware[AgentState[Any], InteractionContext, Any],
):
    tools = ()

    def __init__(
        self,
        bind: Engine | Connection,
        *,
        source: str,
        eligible_sources: tuple[str, ...],
        message_limit: int,
        character_limit: int,
    ) -> None:
        self._bind = bind
        self._source = source
        self._eligible_sources = eligible_sources
        self._message_limit = message_limit
        self._character_limit = character_limit

    def _load(self, *, source: str, conversation_id: str) -> str:
        with Session(self._bind) as session:
            messages = load_recent_personal_context(
                session,
                eligible_sources=self._eligible_sources,
                current_source=source,
                current_conversation_id=conversation_id,
                message_limit=self._message_limit,
                character_limit=self._character_limit,
            )

        return format_recent_personal_context(messages)

    def wrap_model_call(
        self,
        request: ModelRequest[InteractionContext],
        handler: Callable[
            [ModelRequest[InteractionContext]],
            ModelResponse[Any],
        ],
    ) -> ModelCallResult[Any]:
        source, conversation_id = _interaction_identity(
            request.runtime,
            default_source=self._source,
        )
        context = self._load(source=source, conversation_id=conversation_id)
        contextual_request = _append_system_text(request, context) if context else request

        return handler(contextual_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[InteractionContext],
        handler: Callable[
            [ModelRequest[InteractionContext]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelCallResult[Any]:
        source, conversation_id = _interaction_identity(
            request.runtime,
            default_source=self._source,
        )
        context = await asyncio.to_thread(
            self._load,
            source=source,
            conversation_id=conversation_id,
        )
        contextual_request = _append_system_text(request, context) if context else request

        return await handler(contextual_request)
