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
from tool.tts import call_tts


def get_tools() -> list[BaseTool]:
    return [
        call_tts,
        play_audio_queue,
        list_audio_queues,
        get_audio_queue_status,
        stop_audio_queue,
        stop_all_audio_queues,
        set_audio_volume,
        skip_audio_track,
    ]


def get_tool_list() -> str:
    tools = get_tools()
    return "\n".join(
        [f"- {tool.name}: {tool.description.split("\n")[0]}" for tool in tools]
    )
