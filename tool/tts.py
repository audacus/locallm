from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from graph.tts import TTSState, TTSOutputState, tts_compiled


@tool(
    "text_to_speech",
    description="Converts multiple texts to speech and returns the paths to the generated audio files.",
)
def call_tts(
    texts: list[str],
    runtime: ToolRuntime[None, TTSState],
) -> Command:
    if len(texts) == 0:
        tool_message = ToolMessage(
            status="error",
            content="No texts given!",
            tool_call_id=runtime.tool_call_id,
        )

        return Command(update={"messages": [tool_message]})

    result: TTSOutputState = tts_compiled.invoke(
        input={
            "messages": [HumanMessage(content=texts)],
        },
    )

    message = result["messages"][-1].content

    tool_message = ToolMessage(
        content=message,
        tool_call_id=runtime.tool_call_id,
        artifact=result["generations"],
    )

    tool_message.pretty_print()

    return Command(update={"messages": [tool_message]})
