from langchain_core.messages import ToolCall, InvalidToolCall, AIMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.types import Command

from graph.utils import strip_thinking


def output_shaper(state: MessagesState) -> Command:
    """For a clean output of the graph, list all AI messages and tool calls to have an overview of what happened."""
    output_lines: list[str] = []

    def _format_tool_call(tc: ToolCall | InvalidToolCall) -> list[str]:
        lines: list[str] = []
        if tc.get("error"):
            lines.append(f"- Invalid tool call: {tc.get("name", "Tool")}")
            lines.append(f"    Error: {tc.get('error')}")
        else:
            lines.append(f"- Tool call: {tc.get("name", "Tool")}")
        args = tc.get("args")
        if isinstance(args, str):
            lines.append(f"    {args}")
        elif isinstance(args, dict):
            for arg, value in args.items():
                lines.append(f"    {arg}: {value}")

        return lines

    for message in state["messages"]:
        if isinstance(message, AIMessage):
            non_thinking_content = strip_thinking(message.text)
            if len(non_thinking_content) > 0:
                output_lines.append(f"- {non_thinking_content.strip()}")

            # Tool calls.
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_call: ToolCall
                    output_lines.extend(_format_tool_call(tool_call))
            if message.invalid_tool_calls:
                for invalid_tool_call in message.invalid_tool_calls:
                    invalid_tool_call: InvalidToolCall
                    output_lines.extend(_format_tool_call(invalid_tool_call))

        # Tool messages
        if isinstance(message, ToolMessage):
            output_lines.append(f"- Tool result: {message.text}")
            pass

    return Command(update={"output": "\n".join(output_lines)})
