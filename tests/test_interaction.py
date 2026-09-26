import pytest

from m45.interaction import InteractionContext, discord_thread_id


def test_interaction_context_allows_absent_transport_identity() -> None:
    context = InteractionContext()

    assert context.source is None
    assert context.conversation_id is None


@pytest.mark.parametrize(
    ("source", "conversation_id"),
    [
        ("discord", None),
        (None, "123"),
    ],
)
def test_interaction_context_rejects_partial_transport_identity(
    source: str | None,
    conversation_id: str | None,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be provided together",
    ):
        InteractionContext(
            source=source,
            conversation_id=conversation_id,
        )


def test_discord_thread_id_is_stable_and_channel_specific() -> None:
    assert discord_thread_id(123456789012345678) == ("a5c1ec2f-ad44-5c71-a85b-63f7021540fd")
    assert discord_thread_id(123456789012345679) != ("a5c1ec2f-ad44-5c71-a85b-63f7021540fd")


@pytest.mark.parametrize("channel_id", [0, -1])
def test_discord_thread_id_rejects_nonpositive_channel_ids(
    channel_id: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Discord channel ID must be greater than zero",
    ):
        discord_thread_id(channel_id)
