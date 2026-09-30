from typing import Any, cast

from langchain.agents.middleware import AgentState
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from m45.interaction import AGENT_CHAT_UI_SOURCE, InteractionContext
from m45.interaction_instructions import save_interaction_instruction_revision

UPDATE_INTERACTION_INSTRUCTIONS_TOOL_NAME = "update_instructions"


def create_interaction_update_tool(engine: Engine) -> BaseTool:
    @tool(UPDATE_INTERACTION_INSTRUCTIONS_TOOL_NAME)
    def update_instructions(
        expected_revision: int,
        instructions: str,
        runtime: ToolRuntime[InteractionContext, AgentState[Any]],
    ) -> str:
        """Propose a lasting change to Pleiades' standing interaction guidance.

        Use only when the user explicitly asks to change how you generally
        respond across conversations. Do not use for a one-off request, an
        inferred preference, or a personal fact.

        Supply the complete replacement guidance, preserving instructions
        that should remain. Use the revision shown with the current guidance
        as expected_revision. The proposal requires human review before it
        takes effect which is handled by the system.
        """
        context = cast("InteractionContext | None", runtime.context)
        if context is not None and context.source not in (
            None,
            AGENT_CHAT_UI_SOURCE,
        ):
            raise ValueError("instruction updates are available only in Agent Chat UI")

        execution_info = runtime.execution_info
        if execution_info is None or execution_info.thread_id is None:
            raise ValueError("an active LangGraph thread is required")

        conversation_id = (
            context.conversation_id
            if context is not None and context.conversation_id is not None
            else execution_info.thread_id
        )

        user_message = next(
            (
                message
                for message in reversed(runtime.state["messages"])
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        if user_message is None or not user_message.id:
            raise ValueError("the requesting user message needs an ID")
        if not runtime.tool_call_id:
            raise ValueError("the update tool call needs an ID")

        with Session(engine) as session, session.begin():
            revision = save_interaction_instruction_revision(
                session,
                expected_revision=expected_revision,
                instructions=instructions,
                source=AGENT_CHAT_UI_SOURCE,
                conversation_id=conversation_id,
                message_id=user_message.id,
                tool_call_id=runtime.tool_call_id,
            )

        return f"Standing interaction guidance updated to revision {revision}."

    return update_instructions
