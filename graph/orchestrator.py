import base64
import fnmatch
import json
import os
import time
import uuid
from hashlib import md5
from pathlib import Path
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
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import SecretStr

from graph.models import OrchestratorContext
from graph.node.async_tool_node_wrapper import async_tool_node_wrapper
from graph.node.output_shaper import output_shaper
from graph.tools import get_tools, get_tool_list
from prompt.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
from store.references import get_reference_key

load_dotenv()

orchestrator_model = ChatOpenAI(
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


async def call_orchestrator(
    state: OrchestratorState,
    runtime: Runtime[OrchestratorContext],
) -> Command:
    # Build system prompt and prepend to input messages.
    tool_list = get_tool_list()
    system_message = SystemMessage(
        content_blocks=[
            create_text_block(
                text=ORCHESTRATOR_SYSTEM_PROMPT.format(tool_list=tool_list)
            ),
            # TODO: removed TODOs tooling system prompt.
            # create_text_block(text=WRITE_TODOS_SYSTEM_PROMPT),
        ]
    )

    # Handle attached files.
    attachment_dir = Path(os.getenv("ATTACHMENT_DIR"))
    file_counter = 1
    for message in state["messages"]:
        for content_block in message.content_blocks:
            if content_block["type"] == "file" and isinstance(
                content_block["base64"], str
            ):
                # Allow re-use of attached files.
                attachment_hash = md5(
                    content_block["base64"].encode("ascii")
                ).hexdigest()
                file_path = attachment_dir.joinpath(
                    f"{int(time.time())}-{attachment_hash}"
                )

                # Create directory or search for already attached file.
                reuse_attached_file = False
                if not os.path.exists(attachment_dir):
                    os.makedirs(attachment_dir)
                else:
                    for file_name in os.listdir(attachment_dir):
                        if fnmatch.fnmatch(file_name, f"*{attachment_hash}"):
                            file_path = attachment_dir.joinpath(file_name)
                            print("Using already attached file:", file_name)
                            reuse_attached_file = True

                if not reuse_attached_file:
                    with open(file_path, "wb") as f:
                        f.write(
                            base64.b64decode(content_block["base64"].encode("ascii"))
                        )

                attachment_ref = await get_reference_key(
                    runtime.store,
                    runtime.context["user_id"],
                    str(file_path),
                )

                # Transform file content block to text content block.
                lines = [
                    f"File #{file_counter}:",
                    f"  File path reference: {attachment_ref}",
                    f"  MIME type: {content_block["mime_type"]}",
                ]
                content_block["type"] = "text"
                content_block["text"] = "\n".join(lines)
                # Unset file content block specific properties.
                del content_block["base64"]
                del content_block["mime_type"]

                file_counter += 1

    messages = [system_message, *state["messages"]]
    tools = get_tools()
    orchestrator_model_with_tools = orchestrator_model.bind_tools(
        tools=[*tools]
        # TODO: removed TODOs tooling.
        # tools=[write_todos, *tools]
    )
    response: AIMessage = await orchestrator_model_with_tools.ainvoke(input=messages)

    # Special handling for `xLAM 2`.
    # Try to decode the response as JSON array of tool calls.
    if "xlam-2" in response.response_metadata["model_name"].lower():
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
            # `xLAM` can also print non-JSON values; continue.
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

orchestrator_graph.add_node("model", call_orchestrator)
orchestrator_graph.add_node("tools", async_tool_node_wrapper)
orchestrator_graph.add_node("output", output_shaper)
orchestrator_graph.add_conditional_edges(
    "model", tools_condition, {"tools": "tools", END: "output"}
)
orchestrator_graph.add_edge("tools", "model")

orchestrator_graph.set_entry_point("model")
orchestrator_graph.set_finish_point("model")
