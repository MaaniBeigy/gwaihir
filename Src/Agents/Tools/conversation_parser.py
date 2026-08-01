"""Parse Q&A conversation history into a single normalized string."""

from typing import List, Union


def parse_conversation_history(raw_message: Union[str, List[str]]) -> str:
    """Return the conversation as one newline-joined string with empty lines removed.

    Args:
        raw_message: a string or list of strings with `Q:`/`A:` markers.
    """
    if isinstance(raw_message, list):
        all_lines = []
        for entry in raw_message:
            entry = entry.replace("\\n", "\n")
            lines = [line.strip() for line in entry.split("\n") if line.strip()]
            all_lines.extend(lines)
        return "\n".join(all_lines)

    message = raw_message.replace("\\n", "\n")
    lines = [line.strip() for line in message.split("\n") if line.strip()]
    return "\n".join(lines)
