import os

import httpx
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from store.references import get_reference_value, get_reference_key

load_dotenv()

base_url = os.getenv("API_BASE_URL_AUDIO_PLAYBACK")


# --- Pure HTTP logic (no LangGraph awareness) ---


async def _play_audio(queue_name: str, file_paths: list[str], volume: float = 1.0) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/queues/{queue_name}",
            json={"files": file_paths, "volume": volume},
        )
    response.raise_for_status()
    return response.json()


async def _list_queues() -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/queues")
    response.raise_for_status()
    return response.json()


async def _get_queue_status(queue_name: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/queues/{queue_name}")
    response.raise_for_status()
    return response.json()


async def _stop_queue(queue_name: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{base_url}/queues/{queue_name}")
    response.raise_for_status()
    return response.json()


async def _stop_all_queues() -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{base_url}/queues")
    response.raise_for_status()
    return response.json()


async def _set_volume(queue_name: str, volume: float) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{base_url}/queues/{queue_name}/volume",
            json={"volume": volume},
        )
    response.raise_for_status()
    return response.json()


async def _skip_track(queue_name: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/queues/{queue_name}/skip")
    response.raise_for_status()
    return response.json()


# --- Pydantic input schemas ---


class PlayAudioInput(BaseModel):
    queue_name: str = Field(
        description="Unique name for the audio queue (e.g., 'music', 'tts', 'alerts')"
    )
    file_path_refs: list[str] = Field(
        description="List of reference keys (e.g., 'REF_1', 'REF_2') pointing to audio file paths"
    )
    volume: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Playback volume from 0.0 (muted) to 1.0 (full volume)",
    )


class QueueNameInput(BaseModel):
    queue_name: str = Field(description="Name of the audio queue")


class SetVolumeInput(BaseModel):
    queue_name: str = Field(description="Name of the audio queue")
    volume: float = Field(
        ge=0.0,
        le=1.0,
        description="Volume level from 0.0 (muted) to 1.0 (full volume)",
    )


# --- @tool wrappers (handle references, ToolMessage, Command) ---


@tool(
    "play_audio_queue",
    description="Play audio files by adding them to a named playback queue. Accepts reference keys to audio file paths.",
    args_schema=PlayAudioInput,
)
async def play_audio_queue(
    queue_name: str,
    file_path_refs: list[str],
    volume: float,
    runtime: ToolRuntime,
) -> Command:
    if len(file_path_refs) == 0:
        tool_error_message = ToolMessage(
            content="No file path references given!",
            status="error",
            tool_call_id=runtime.tool_call_id,
        )
        tool_error_message.pretty_print()
        return Command(update={"messages": [tool_error_message]})

    # Resolve reference keys to actual file paths.
    file_paths: list[str] = []
    for ref in file_path_refs:
        path = await get_reference_value(
            runtime.store,
            runtime.config["configurable"]["context"]["user_id"],
            ref,
        )
        if path is None:
            tool_message = ToolMessage(
                content=f"Reference '{ref}' not found.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
            tool_message.pretty_print()
            return Command(update={"messages": [tool_message]})
        file_paths.append(path)

    status = await _play_audio(queue_name, file_paths, volume)

    message_lines = [f"Playing on queue '{queue_name}':"]
    for ref, path in zip(file_path_refs, file_paths):
        message_lines.append(f"  - {ref}")
    if status.get("current_file"):
        message_lines.append(f"Now playing: {status['current_file']}")

    tool_message = ToolMessage(
        content="\n".join(message_lines),
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "list_audio_queues",
    description="List all active audio playback queues with their status.",
)
async def list_audio_queues(
    runtime: ToolRuntime,
) -> Command:
    queues = await _list_queues()

    if not queues:
        content = "No active audio queues."
    else:
        lines = ["Active audio queues:"]
        for q in queues:
            playing = "playing" if q["is_playing"] else "idle"
            lines.append(
                f"  - {q['name']}: volume={q['volume']}, {playing}, {q['file_count']} file(s)"
            )
        content = "\n".join(lines)

    tool_message = ToolMessage(
        content=content,
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "get_audio_queue_status",
    description="Get the current status of an audio playback queue.",
    args_schema=QueueNameInput,
)
async def get_audio_queue_status(
    queue_name: str,
    runtime: ToolRuntime,
) -> Command:
    try:
        status = await _get_queue_status(queue_name)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            tool_error_message = ToolMessage(
                content=f"Queue '{queue_name}' not found.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
            tool_error_message.pretty_print()
            return Command(update={"messages": [tool_error_message]})
        raise

    playing = "playing" if status["is_playing"] else "idle"
    current_file_ref = await get_reference_key(
        runtime.store,
        runtime.config["configurable"]["context"]["user_id"],
        status["current_file"],
    )
    lines = [
        f"Queue '{queue_name}': {playing}",
        f"  Current playing file ref: {current_file_ref or 'none'}",
        f"  Position: {status['current_position']:.1f}s / {status['current_duration']:.1f}s",
        f"  Remaining: {len(status['remaining_files'])} file(s)",
        f"  Volume: {status['volume']}",
    ]

    tool_message = ToolMessage(
        content="\n".join(lines),
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "stop_audio_queue",
    description="Stop audio playback and remove a queue.",
    args_schema=QueueNameInput,
)
async def stop_audio_queue(
    queue_name: str,
    runtime: ToolRuntime,
) -> Command:
    try:
        result = await _stop_queue(queue_name)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            tool_error_message = ToolMessage(
                content=f"Queue '{queue_name}' not found.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
            tool_error_message.pretty_print()
            return Command(update={"messages": [tool_error_message]})
        raise

    tool_message = ToolMessage(
        content=result.get("message", f"Queue '{queue_name}' stopped."),
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "stop_all_audio_queues",
    description="Stop all audio playback and remove all queues.",
)
async def stop_all_audio_queues(
    runtime: ToolRuntime,
) -> Command:
    result = await _stop_all_queues()

    tool_message = ToolMessage(
        content=result.get("message", "All audio queues stopped."),
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "set_audio_volume",
    description="Set the volume level for an audio playback queue.",
    args_schema=SetVolumeInput,
)
async def set_audio_volume(
    queue_name: str,
    volume: float,
    runtime: ToolRuntime,
) -> Command:
    try:
        await _set_volume(queue_name, volume)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            tool_error_message = ToolMessage(
                content=f"Queue '{queue_name}' not found.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
            tool_error_message.pretty_print()
            return Command(update={"messages": [tool_error_message]})
        raise

    tool_message = ToolMessage(
        content=f"Volume for queue '{queue_name}' set to {volume}.",
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})


@tool(
    "skip_audio_track",
    description="Skip to the next audio file in a queue.",
    args_schema=QueueNameInput,
)
async def skip_audio_track(
    queue_name: str,
    runtime: ToolRuntime,
) -> Command:
    try:
        await _skip_track(queue_name)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            tool_error_message = ToolMessage(
                content=f"Queue '{queue_name}' not found.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
            tool_error_message.pretty_print()
            return Command(update={"messages": [tool_error_message]})
        raise

    tool_message = ToolMessage(
        content=f"Skipped track in queue '{queue_name}'.",
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})
