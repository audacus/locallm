from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from simpleaudio import WaveObject


# Other libraries: https://realpython.com/playing-and-recording-sound-python/#playing-audio-files
@tool(
    "play_audio",
    description="Play audio file at given path.",
)
def play_audio(
    file_path: str,
    runtime: ToolRuntime[None, MessagesState],
) -> Command:
    wave_obj = WaveObject.from_wave_file(file_path)
    play_obj = wave_obj.play()
    print(play_obj)

    tool_message = ToolMessage(
        content=f"Playing audio file: {file_path}",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        update={
            "messages": [tool_message],
        }
    )
