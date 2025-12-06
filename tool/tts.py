from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from graph.tts import TTSState, TTSOutputState, tts_compiled


@tool(
    "text_to_speech",
    description="Convert text to speech and return the path to the audio file.",
)
def call_tts(
    text: str,
    runtime: ToolRuntime[None, TTSState],
) -> Command:
    result: TTSOutputState = tts_compiled.invoke(
        input={
            "messages": [HumanMessage(content=text)],
        },
    )

    message = result["messages"][-1].content
    file_path = result["file_path"]
    cached = result["cached"]
    generation = result["generation"]

    tool_message = ToolMessage(
        content=message,
        tool_call_id=runtime.tool_call_id,
        file_path=file_path,
        cached=cached,
        generation=generation,
    )

    tool_message.pretty_print()

    return Command(
        update={"messages": [tool_message]},
    )
