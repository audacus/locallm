import re


def strip_thinking(content: str) -> str:
    """Strip `<think>` from message content."""
    regex = re.compile(r"<think>.*?</think>", re.DOTALL)
    non_thinking_content = re.sub(regex, "", content).strip()
    return non_thinking_content
