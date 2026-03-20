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
from tool.language import translate_text
from tool.stt import call_transcribe_audio
from tool.tts import convert_text_to_speech


def get_tools() -> list[BaseTool]:
    return [
        convert_text_to_speech,
        call_transcribe_audio,
        get_audio_queue_status,
        get_mime_type,
        list_audio_queues,
        play_audio_queue,
        read_file,
        set_audio_volume,
        skip_audio_track,
        sleep,
        stop_all_audio_queues,
        stop_audio_queue,
        translate_text,
    ]


def get_tool_list() -> str:
    tools = get_tools()
    return "\n".join(
        [f"- {tool.name}: {tool.description.split("\n")[0]}" for tool in tools]
    )
