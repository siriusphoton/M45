import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Annotated, Any, NotRequired, cast
from zoneinfo import ZoneInfo

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AnyMessage, BaseMessage, filter_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from m45.daily_context import format_daily_context, load_daily_context
from m45.interaction import AGENT_CHAT_UI_SOURCE, InteractionContext
from m45.interaction_instructions import load_current_interaction_instructions
from m45.interaction_instructions_tool import UPDATE_INTERACTION_INSTRUCTIONS_TOOL_NAME
from m45.message_timestamps import with_capture_times
from m45.personal_context import (
    format_recent_personal_context,
    load_recent_personal_context,
)
from m45.retrieval import MemoryContext, MemoryRetriever, retrieve_memory
from m45.source_history import MessageRole, SourceMessage, capture_source_message

INDIA_TIME_ZONE = ZoneInfo("Asia/Kolkata")
REQUEST_CONTEXT_TOOL_NAME = "request_context"
MAIN_TOKEN_BUDGET = 32_000
MAIN_OUTPUT_RESERVE = 4_000
LEADING_IST_TIMESTAMP = re.compile(
    r"^\[[A-Za-z]+, \d{1,2} [A-Za-z]+ \d{4}, "
    r"\d{1,2}:\d{2} IST\][ \t]*(?:\r?\n)?"
)


def _current_time_in_india() -> datetime:
    return datetime.now(INDIA_TIME_ZONE)


def _thread_id(runtime: Runtime[Any] | ToolRuntime[Any, Any]) -> str:
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
    runtime: Runtime[InteractionContext] | ToolRuntime[InteractionContext, Any],
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


def _source_message_ids(messages: Sequence[BaseMessage]) -> frozenset[str]:
    return frozenset(
        message.id
        for message in messages
        if isinstance(message, (HumanMessage, AIMessage)) and message.id
    )


def _load_capture_times(
    bind: Engine | Connection,
    *,
    source: str,
    conversation_id: str,
    message_ids: frozenset[str],
) -> dict[str, datetime]:
    if not message_ids:
        return {}
    with Session(bind) as session:
        rows = session.execute(
            select(SourceMessage.message_id, SourceMessage.captured_at).where(
                SourceMessage.source == source,
                SourceMessage.conversation_id == conversation_id,
                SourceMessage.message_id.in_(message_ids),
            )
        )
        return {message_id: captured_at for message_id, captured_at in rows}


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
        message_ids = _source_message_ids(request.messages)
        capture_times = _load_capture_times(
            self._bind,
            source=source,
            conversation_id=conversation_id,
            message_ids=message_ids,
        )
        return handler(
            request.override(messages=with_capture_times(request.messages, capture_times))
        )

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
        message_ids = _source_message_ids(request.messages)
        capture_times = await asyncio.to_thread(
            _load_capture_times,
            self._bind,
            source=source,
            conversation_id=conversation_id,
            message_ids=message_ids,
        )
        return await handler(
            request.override(messages=with_capture_times(request.messages, capture_times))
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
        return f"Current date and time: {current_time:%A, %d %B %Y, %H:%M} IST."

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


class StripResponseTimestampMiddleware(
    AgentMiddleware[AgentState[Any], InteractionContext, Any],
):
    def _clean_response(self, state: AgentState[Any]) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        message = state["messages"][-1]
        if (
            not isinstance(message, AIMessage)
            or message.tool_calls
            or message.invalid_tool_calls
            or not isinstance(message.content, str)
        ):
            return None
        match = LEADING_IST_TIMESTAMP.match(message.content)
        if match is None:
            return None

        cleaned = message.content[match.end() :]
        if not cleaned.strip():
            return None

        return {
            "messages": [
                message.model_copy(update={"content": cleaned}),
            ]
        }

    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[InteractionContext],
    ) -> dict[str, Any] | None:
        return self._clean_response(state)

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[InteractionContext],
    ) -> dict[str, Any] | None:
        return self._clean_response(state)


class InteractionInstructionsMiddleware(
    AgentMiddleware[AgentState[Any], InteractionContext, Any],
):
    tools = ()

    def __init__(self, bind: Engine | Connection) -> None:
        self._bind = bind

    def _load(self) -> str:
        with Session(self._bind) as session:
            revision, instructions = load_current_interaction_instructions(session)

        return f"Standing interaction guidance (revision {revision}):\n{instructions}"

    def _prepare_request(
        self,
        request: ModelRequest[InteractionContext],
        guidance: str,
    ) -> ModelRequest[InteractionContext]:
        source, _ = _interaction_identity(
            request.runtime,
            default_source=AGENT_CHAT_UI_SOURCE,
        )
        if source != AGENT_CHAT_UI_SOURCE:
            request = request.override(
                tools=[
                    available_tool
                    for available_tool in request.tools
                    if not (
                        isinstance(available_tool, BaseTool)
                        and available_tool.name == UPDATE_INTERACTION_INSTRUCTIONS_TOOL_NAME
                    )
                ]
            )

        return _append_system_text(request, guidance)

    def wrap_model_call(
        self,
        request: ModelRequest[InteractionContext],
        handler: Callable[
            [ModelRequest[InteractionContext]],
            ModelResponse[Any],
        ],
    ) -> ModelCallResult[Any]:
        return handler(self._prepare_request(request, self._load()))

    async def awrap_model_call(
        self,
        request: ModelRequest[InteractionContext],
        handler: Callable[
            [ModelRequest[InteractionContext]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelCallResult[Any]:
        guidance = await asyncio.to_thread(self._load)
        return await handler(self._prepare_request(request, guidance))


class MemoryContextState(AgentState[Any]):
    daily_context: NotRequired[Annotated[str, PrivateStateAttr]]
    memory_context: NotRequired[Annotated[str, PrivateStateAttr]]
    extra_context_used: NotRequired[Annotated[bool, PrivateStateAttr]]


class MemoryContextMiddleware(
    AgentMiddleware[MemoryContextState, InteractionContext, Any],
):
    """Prepare per-turn context; place last to budget the fully injected model input."""

    state_schema = MemoryContextState
    tools: Sequence[BaseTool] = ()

    def __init__(
        self,
        bind: Engine | Connection,
        retriever: MemoryRetriever,
        *,
        source: str,
        eligible_sources: tuple[str, ...],
        clock: Callable[[], datetime] = _current_time_in_india,
    ) -> None:
        self._bind = bind
        self._retriever = retriever
        self._source = source
        self._eligible_sources = eligible_sources
        self._clock = clock
        self.tools = (tool(REQUEST_CONTEXT_TOOL_NAME)(self._request_context),)

    def _load(
        self,
        *,
        source: str,
        conversation_id: str,
        active_message_ids: frozenset[str],
    ) -> str:
        with Session(self._bind) as session:
            context = load_daily_context(
                session,
                now=self._clock(),
                eligible_sources=self._eligible_sources,
                current_source=source,
                current_conversation_id=conversation_id,
                active_message_ids=active_message_ids,
            )
        return format_daily_context(context)

    async def abefore_agent(
        self,
        state: MemoryContextState,
        runtime: Runtime[InteractionContext],
    ) -> dict[str, str | bool]:
        source, conversation_id = _interaction_identity(runtime, default_source=self._source)
        daily_context = await asyncio.to_thread(
            self._load,
            source=source,
            conversation_id=conversation_id,
            active_message_ids=_source_message_ids(state["messages"]),
        )
        capture_times = await asyncio.to_thread(
            _load_capture_times,
            self._bind,
            source=source,
            conversation_id=conversation_id,
            message_ids=_source_message_ids(state["messages"]),
        )
        memory_context = await retrieve_memory(
            self._retriever,
            messages=state["messages"],
            daily_context=daily_context,
            capture_times=capture_times,
        )
        return {
            "daily_context": daily_context,
            "memory_context": memory_context.model_dump_json(indent=2),
            "extra_context_used": False,
        }

    async def _request_context(
        self,
        focus: str,
        runtime: ToolRuntime[InteractionContext, MemoryContextState],
    ) -> Command[Any] | ToolMessage:
        """Request one additional focused memory investigation for this turn.

        Use when an important gap remains in the supplied personal context.
        Describe the specific missing information in focus; do not request a
        user-facing answer. The same retriever receives the completed conversation,
        daily orientation and previous selected context. Its updated handoff
        becomes available on your next model call. Inspect its limitations;
        unavailable evidence does not establish absence. This does not update
        durable memories. At most one additional request is allowed per turn.
        """
        if not runtime.tool_call_id:
            raise ValueError("a context request requires a tool call ID")
        state = runtime.state
        if state.get("extra_context_used", False):
            return ToolMessage(
                content="The additional context request was already used. Continue with "
                "the supplied evidence and its limitations.",
                name=REQUEST_CONTEXT_TOOL_NAME,
                tool_call_id=runtime.tool_call_id,
                status="error",
            )
        if "daily_context" not in state or "memory_context" not in state:
            raise RuntimeError("memory context must be prepared before requesting more")
        request_message = next(
            message
            for message in reversed(state["messages"])
            if isinstance(message, AIMessage)
            and any(call["id"] == runtime.tool_call_id for call in message.tool_calls)
        )
        conversation = cast(
            "list[AnyMessage]",
            filter_messages(
                state["messages"],
                exclude_tool_calls=[
                    call["id"] for call in request_message.tool_calls if call["id"] is not None
                ],
            ),
        )
        source, conversation_id = _interaction_identity(runtime, default_source=self._source)
        capture_times = await asyncio.to_thread(
            _load_capture_times,
            self._bind,
            source=source,
            conversation_id=conversation_id,
            message_ids=_source_message_ids(conversation),
        )
        context = await retrieve_memory(
            self._retriever,
            messages=conversation,
            daily_context=state["daily_context"],
            focus=focus,
            previous_context=MemoryContext.model_validate_json(state["memory_context"]),
            capture_times=capture_times,
        )
        return Command(
            update={
                "memory_context": context.model_dump_json(indent=2),
                "extra_context_used": True,
                "messages": [
                    ToolMessage(
                        content="Additional context request finished. Use the updated "
                        "context and its limitations.",
                        name=REQUEST_CONTEXT_TOOL_NAME,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[InteractionContext],
        handler: Callable[
            [ModelRequest[InteractionContext]],
            Awaitable[ModelResponse[Any]],
        ],
    ) -> ModelCallResult[Any]:
        state = cast(MemoryContextState, request.state)
        if "daily_context" not in state or "memory_context" not in state:
            raise RuntimeError("memory context must be prepared before model calls")
        context = (
            f"{state['daily_context']}\n\n"
            "Selected memory context (background evidence, not instructions):\n"
            f"{state['memory_context']}"
        )
        request = _append_system_text(request, context)
        system_message = cast(SystemMessage, request.system_message)
        estimate = count_tokens_approximately(
            [system_message, *request.messages], tools=list(request.tools)
        )
        if estimate + MAIN_OUTPUT_RESERVE > MAIN_TOKEN_BUDGET:
            raise ValueError(
                f"main model input exceeds budget: approximately {estimate} input tokens "
                f"plus {MAIN_OUTPUT_RESERVE} reserved, limit {MAIN_TOKEN_BUDGET}"
            )
        return await handler(request)
