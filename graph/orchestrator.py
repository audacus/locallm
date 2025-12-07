import json
import os
import re
import uuid
from typing import TypedDict, Annotated, NotRequired

from dotenv import load_dotenv
from langchain.agents.middleware.todo import (
    Todo,
    WRITE_TODOS_SYSTEM_PROMPT,
    write_todos,
)
from langchain.agents.middleware.types import OmitFromInput
from langchain_core.messages import (
    ToolMessage,
    ToolCall,
    InvalidToolCall,
    AIMessage,
    SystemMessage,
    HumanMessage,
)
from langchain_core.messages.content import create_text_block
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import SecretStr

from graph.tools import TOOLS
from prompt.input_enricher import INPUT_ENRICHER_PROMPT
from prompt.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

load_dotenv()

TOOL_LIST = "\n".join([f"- {tool.name}: {tool.description}" for tool in TOOLS])

input_enricher_model = ChatOpenAI(
    async_client=False,
    base_url=os.getenv("API_BASE_URL_ORCHESTRATOR"),
    api_key=SecretStr("none"),
    model=os.getenv("MODEL_INPUT_ENRICHER"),
    streaming=False,
    temperature=0,
    max_tokens=4096,
)

orchestrator_model = ChatOpenAI(
    async_client=True,
    base_url=os.getenv("API_BASE_URL_ORCHESTRATOR"),
    api_key=SecretStr("none"),
    model=os.getenv("MODEL_ORCHESTRATOR"),
    streaming=True,
    temperature=0,
    max_tokens=4096,
)
orchestrator_model_with_tools = orchestrator_model.bind_tools(
    tools=[write_todos, *TOOLS]
)


class OrchestratorState(MessagesState):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]


class OrchestratorOutputState(TypedDict):
    output: str


def strip_thinking(content: str) -> str:
    """Strip `<think>` from message content."""
    regex = re.compile(r"<think>.*?</think>", re.DOTALL)
    non_thinking_content = re.sub(regex, "", content).strip()
    return non_thinking_content


def input_enricher(state: OrchestratorState) -> Command:
    input_message = state["messages"][-1]

    input_message.pretty_print()

    # Prompt an LLM separately to enrich the original input.
    message = HumanMessage(
        content=INPUT_ENRICHER_PROMPT.format(
            input=input_message.text.strip(),
            tool_list=TOOL_LIST,
        )
    )
    response: AIMessage = input_enricher_model.invoke(input=[message])

    enriched_input_message = HumanMessage(content=strip_thinking(response.content))

    enriched_input_message.pretty_print()

    # Append the enriched input prompt.
    return Command(update={"messages": [input_message, enriched_input_message]})


def call_orchestrator(state: OrchestratorState) -> Command:
    # Build system prompt and prepend to input messages.
    system_message = SystemMessage(
        content_blocks=[
            create_text_block(
                text=ORCHESTRATOR_SYSTEM_PROMPT.format(tool_list=TOOL_LIST)
            ),
            create_text_block(text=WRITE_TODOS_SYSTEM_PROMPT),
        ]
    )

    messages = [system_message, *state["messages"]]
    response: AIMessage = orchestrator_model_with_tools.invoke(input=messages)

    # Special handling for `xLAM 2`.
    # Try to decode the response as JSON array of tool calls.
    if "xLAM-2" in response.response_metadata["model_name"]:
        tool_calls: list[ToolCall] = []
        try:
            obj = json.loads(response.text)
            if isinstance(obj, list):
                for item in obj:
                    if "name" in item and "arguments" in item:
                        tool_calls.append(
                            ToolCall(
                                name=item["name"],
                                args=item["arguments"],
                                id=str(uuid.uuid4()),
                            )
                        )
        except json.decoder.JSONDecodeError:
            print("Invalid JSON:", response.text)
            raise

        # Add or overwrite on the original response.
        if response.tool_calls:
            response.tool_calls.extend(tool_calls)
        else:
            response.tool_calls = tool_calls

    # Ensure there is a tool call ID set.
    # It is set to `None` by `mlx_lm.server.APIHandler.generate_response` -> `parse_function()`
    if isinstance(response, AIMessage) and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_call: ToolCall
            if not tool_call["id"]:
                tool_call["id"] = str(uuid.uuid4())

    response.pretty_print()

    return Command(update={"messages": [response]})


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


orchestrator_graph = StateGraph(
    state_schema=OrchestratorState,
    output_schema=OrchestratorOutputState,
)
orchestrator_graph.add_node("enricher", input_enricher)
orchestrator_graph.add_node("model", call_orchestrator)
orchestrator_graph.add_node(
    "tools",
    ToolNode(
        TOOLS,
        handle_tool_errors=False,
    ),
)
orchestrator_graph.add_node("output", output_shaper)

orchestrator_graph.add_edge("enricher", "model")
orchestrator_graph.add_conditional_edges(
    "model", tools_condition, {"tools": "tools", END: "output"}
)
orchestrator_graph.add_edge("tools", "model")

orchestrator_graph.set_entry_point("enricher")
orchestrator_graph.set_finish_point("model")

orchestrator_compiled = orchestrator_graph.compile()
