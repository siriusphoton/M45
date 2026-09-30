from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from m45.interaction import AGENT_CHAT_UI_SOURCE
from m45.interaction_instructions import (
    InteractionInstructionRevision,
    load_current_interaction_instructions,
    save_interaction_instruction_revision,
)


def test_approved_instruction_revision_is_idempotent_and_rejects_stale_edits(
    session: Session,
) -> None:
    previous_revision, _ = load_current_interaction_instructions(session)
    tool_call_id = f"test-instruction-update-{uuid4()}"
    provenance = {
        "source": AGENT_CHAT_UI_SOURCE,
        "conversation_id": "test-interaction-thread",
        "message_id": "test-user-request",
        "tool_call_id": tool_call_id,
    }
    instructions = "Keep answers brief unless I ask for detail."

    revision = save_interaction_instruction_revision(
        session,
        expected_revision=previous_revision,
        instructions=instructions,
        **provenance,
    )

    assert revision == previous_revision + 1
    assert load_current_interaction_instructions(session) == (revision, instructions)
    saved = session.get(InteractionInstructionRevision, revision)
    assert saved is not None
    assert (
        saved.source,
        saved.conversation_id,
        saved.message_id,
        saved.tool_call_id,
    ) == (
        AGENT_CHAT_UI_SOURCE,
        "test-interaction-thread",
        "test-user-request",
        tool_call_id,
    )

    assert (
        save_interaction_instruction_revision(
            session,
            expected_revision=previous_revision,
            instructions=instructions,
            **provenance,
        )
        == revision
    )

    with pytest.raises(ValueError, match="interaction instructions changed"):
        save_interaction_instruction_revision(
            session,
            expected_revision=previous_revision,
            instructions="Use a different response style.",
            source=AGENT_CHAT_UI_SOURCE,
            conversation_id="test-interaction-thread",
            message_id="another-user-request",
            tool_call_id=f"test-stale-update-{uuid4()}",
        )

    assert load_current_interaction_instructions(session) == (revision, instructions)
