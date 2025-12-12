import json
import os
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
    ToolCall,
    InvalidToolCall,
    AIMessage,
    SystemMessage,
)
from langchain_core.messages.content import create_text_block
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.types import Command
from pydantic import SecretStr

from graph.node.async_tool_node_wrapper import async_tool_node_wrapper
from graph.node.complexity_switcher import complexity_switcher
from graph.node.input_enricher import input_enricher
from graph.node.output_shaper import output_shaper
from graph.tools import get_tools, get_tool_list
from prompt.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT

load_dotenv()

orchestrator_model = ChatOpenAI(
    async_client=True,
    base_url=os.getenv("API_BASE_URL_MLX_LM"),
    api_key=SecretStr("none"),
    model=os.getenv("MODEL_ORCHESTRATOR"),
    streaming=True,
    temperature=0,
    max_tokens=4096,
)


class OrchestratorState(MessagesState):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]


class OrchestratorOutputState(TypedDict):
    output: str


async def call_orchestrator(state: OrchestratorState) -> Command:
    # Build system prompt and prepend to input messages.
    tool_list = await get_tool_list()
    system_message = SystemMessage(
        content_blocks=[
            create_text_block(
                text=ORCHESTRATOR_SYSTEM_PROMPT.format(tool_list=tool_list)
            ),
            create_text_block(text=WRITE_TODOS_SYSTEM_PROMPT),
        ]
    )

    messages = [system_message, *state["messages"]]
    tools = await get_tools()
    orchestrator_model_with_tools = orchestrator_model.bind_tools(
        tools=[write_todos, *tools]
    )
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
            # `xLAM` can also print non-JSOn values; continue.
            pass

        # Add or overwrite on the original response.
        if response.tool_calls:
            response.tool_calls.extend(tool_calls)
        else:
            response.tool_calls = tool_calls

    # Ensure there is a tool call ID set.
    # It is set to `None` by `mlx_lm.server.APIHandler.generate_response` -> `parse_function()`
    if isinstance(response, AIMessage):
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_call: ToolCall
                if not tool_call["id"]:
                    tool_call["id"] = str(uuid.uuid4())
        if response.invalid_tool_calls:
            for invalid_tool_call in response.invalid_tool_calls:
                invalid_tool_call: InvalidToolCall
                if not invalid_tool_call["id"]:
                    invalid_tool_call["id"] = str(uuid.uuid4())

    response.pretty_print()

    return Command(update={"messages": [response]})


orchestrator_graph = StateGraph(
    state_schema=OrchestratorState,
    output_schema=OrchestratorOutputState,
)
orchestrator_graph.add_node("switch", complexity_switcher)
orchestrator_graph.add_node("enricher", input_enricher)

orchestrator_graph.add_node("model", call_orchestrator)
orchestrator_graph.add_node("tools", async_tool_node_wrapper)
orchestrator_graph.add_node("output", output_shaper)
orchestrator_graph.add_edge("enricher", "model")
orchestrator_graph.add_conditional_edges(
    "model", tools_condition, {"tools": "tools", END: "output"}
)
orchestrator_graph.add_edge("tools", "model")

orchestrator_graph.set_entry_point("model")
orchestrator_graph.set_finish_point("model")

orchestrator_compiled = orchestrator_graph.compile()
