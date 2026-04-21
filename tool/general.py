import asyncio

import magic
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from store.references import get_reference_value


class SleepInput(BaseModel):
    seconds: int = Field(description="The number of seconds to sleep")


class ReadFileInput(BaseModel):
    file_path_ref: str = Field(
        description="Reference key (e.g. 'REF_1', 'REF_2') pointing to the path of the file.",
    )


class GetMIMETypeInput(BaseModel):
    file_path_ref: str = Field(
        description="Reference key (e.g. 'REF_1', 'REF_2') pointing to the path of the file.",
    )


@tool(
    "sleep",
    description="sleep, pause, wait, timeout",
    args_schema=SleepInput,
)
async def sleep(
    seconds: int,
    runtime: ToolRuntime,
) -> Command:
    await asyncio.sleep(seconds)

    tool_message = ToolMessage(
        content=f"Done. Slept for {seconds} seconds.",
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "read_file",
    description="Returns the content of a file as text.",
    args_schema=ReadFileInput,
)
async def read_file(
    file_path_ref: str,
    runtime: ToolRuntime,
) -> Command:
    file_path = await get_reference_value(
        runtime.store,
        runtime.config["configurable"]["context"]["user_id"],
        file_path_ref,
    )
    if file_path is None:
        tool_error_message = ToolMessage(
            content=f"Reference '{file_path_ref}' not found.",
            status="error",
            tool_call_id=runtime.tool_call_id,
        )
        tool_error_message.pretty_print()
        return Command(update={"messages": [tool_error_message]})

    with open(file_path, "r") as file:
        content = file.read()

    tool_message = ToolMessage(
        content=content,
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "get_mime_type",
    description="Returns the MIME type of a file.",
    args_schema=GetMIMETypeInput,
)
async def get_mime_type(
    file_path_ref: str,
    runtime: ToolRuntime,
) -> Command:
    file_path = await get_reference_value(
        runtime.store,
        runtime.config["configurable"]["context"]["user_id"],
        file_path_ref,
    )
    if file_path is None:
        tool_error_message = ToolMessage(
            content=f"Reference '{file_path_ref}' not found.",
            status="error",
            tool_call_id=runtime.tool_call_id,
        )
        tool_error_message.pretty_print()
        return Command(update={"messages": [tool_error_message]})

    mime_type = magic.from_file(file_path, mime=True)

    tool_message = ToolMessage(
        content=f"MIME type of {file_path_ref}: {mime_type}",
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})
