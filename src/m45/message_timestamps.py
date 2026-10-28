from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


def with_capture_times(
    messages: Sequence[AnyMessage], capture_times: dict[str, datetime]
) -> list[AnyMessage]:
    """Result model input copies with known source capture times; leave originals intact."""
    result = list(messages)
    for index, message in enumerate(result):
        if isinstance(message, (HumanMessage, AIMessage)) and message.id in capture_times:
            local_time = capture_times[message.id].astimezone(ZoneInfo("Asia/Kolkata"))
            label = f"[{local_time:%A, %d %B %Y, %H:%M} IST]\n"
            content = message.content
            if isinstance(content, str):
                content = label + content
            else:
                content = [{"type": "text", "text": label}, *content]
            result[index] = message.model_copy(update={"content": content})
    return result
