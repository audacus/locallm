import os
import re
import uuid
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, ToolCall, InvalidToolCall, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import SecretStr

from graph.tools import TOOLS

load_dotenv()

orchestrator_chat_model = ChatOpenAI(
    async_client=True,
    base_url=os.getenv("API_BASE_URL_ORCHESTRATOR"),
    api_key=SecretStr("none"),
    model=os.getenv("MODEL_ORCHESTRATOR"),
    streaming=True,
    temperature=0.0,
    max_tokens=4096,
)
orchestrator_chat_model_with_tools = orchestrator_chat_model.bind_tools(tools=TOOLS)


class OrchestratorState(MessagesState):
    pass


class OrchestratorOutputState(TypedDict):
    output: str


def call_orchestrator(state: OrchestratorState) -> Command:
    response: AIMessage = orchestrator_chat_model_with_tools.invoke(
        input=state["messages"]
    )

    # Ensure there is a tool call ID set.
    # It is set to `None` by `mlx_lm.server.APIHandler.generate_response` -> `parse_function()`
    if isinstance(response, AIMessage) and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_call: ToolCall
            if not tool_call["id"]:
                tool_call["id"] = str(uuid.uuid4())

    response.pretty_print()

    return Command(
        update={
            "messages": [response],
        }
    )


def output_shaper(state: OrchestratorState) -> Command:
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
            # Strip `<think>` from message content.
            regex = re.compile(r"<think>.*?</think>", re.DOTALL)
            non_thinking_content = re.sub(regex, "", message.text).strip()
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
            output_lines.append(f"- Tool result: {message.content}")
            pass

    return Command(
        update={
            "output": "\n".join(output_lines),
        }
    )


orchestrator_graph = StateGraph(
    state_schema=OrchestratorState,
    output_schema=OrchestratorOutputState,
)
orchestrator_graph.add_node("model", call_orchestrator)
orchestrator_graph.add_node(
    "tools",
    ToolNode(
        TOOLS,
        handle_tool_errors=False,
    ),
)
orchestrator_graph.add_node("output", output_shaper)

orchestrator_graph.add_conditional_edges(
    "model", tools_condition, {"tools": "tools", END: "output"}
)
orchestrator_graph.add_edge("tools", "model")

orchestrator_graph.set_entry_point("model")
orchestrator_graph.set_finish_point("model")

orchestrator_compiled = orchestrator_graph.compile()
