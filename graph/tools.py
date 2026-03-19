from langchain_core.tools import BaseTool

from tool.audio_playback import (
    get_audio_queue_status,
    list_audio_queues,
    play_audio_queue,
    set_audio_volume,
    skip_audio_track,
    stop_all_audio_queues,
    stop_audio_queue,
)
from tool.general import sleep, read_file, get_mime_type
from tool.stt import call_transcribe_audio
from tool.tts import call_convert_text_to_speech


def get_tools() -> list[BaseTool]:
    return [
        call_transcribe_audio,
        call_convert_text_to_speech,
        get_audio_queue_status,
        # TODO: don't add unnecessary tool.
        # get_mime_type,
        list_audio_queues,
        play_audio_queue,
        read_file,
        set_audio_volume,
        skip_audio_track,
        sleep,
        stop_all_audio_queues,
        stop_audio_queue,
    ]


def get_tool_list() -> str:
    tools = get_tools()
    return "\n".join(
        [f"- {tool.name}: {tool.description.split("\n")[0]}" for tool in tools]
    )
