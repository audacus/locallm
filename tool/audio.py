import os

import sounddevice
import soundfile
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field


class AudioInput(BaseModel):
    file_paths: list[str] = Field(description="Paths to audio files.")


# Other libraries: https://realpython.com/playing-and-recording-sound-python/#playing-audio-files
@tool(
    "play_audio",
    description="Plays a list of audio files.",
    args_schema=AudioInput,
)
def play_audio(
    file_paths: list[str],
    runtime: ToolRuntime[None, MessagesState],
) -> Command:
    if len(file_paths) == 0:
        tool_error_message = ToolMessage(
            status="error",
            content="No file paths given!",
            tool_call_id=runtime.tool_call_id,
        )

        tool_error_message.pretty_print()

        return Command(update={"messages": [tool_error_message]})

    good_paths: list[str] = []
    bad_paths: list[str] = []
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            bad_paths.append(file_path)
        else:
            good_paths.append(file_path)

            data, samplerate = soundfile.read(file_path)
            sounddevice.play(data, samplerate)
            sounddevice.wait()

    message_lines: list[str] = []
    if len(good_paths) > 0:
        message_lines.append("Playing audio files:")
        for good_path in good_paths:
            message_lines.append(f"  - {good_path}")

    if len(bad_paths) > 0:
        message_lines.append("Invalid file paths:")
        for bad_path in bad_paths:
            message_lines.append(f"  - {bad_path}")

    if not good_paths and not bad_paths:
        message_lines.append("No audio files playing.")

    tool_error_message = ToolMessage(
        content="\n".join(message_lines),
        tool_call_id=runtime.tool_call_id,
    )

    tool_error_message.pretty_print()

    return Command(update={"messages": [tool_error_message]})
