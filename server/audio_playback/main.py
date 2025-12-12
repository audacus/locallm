"""Audio Playback Service - FastAPI application with multiple named audio queues."""

import logging
import os
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from models import QueueCreate, QueueInfo, QueueStatus, VolumeUpdate
from queue_manager import QueueManager
from state import StatePersistence

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
queue_manager = QueueManager()
state_persistence = StatePersistence()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage application lifecycle."""
    # Startup: load state and restore queues
    state_persistence.set_state_callback(queue_manager.get_state)
    saved_state = await state_persistence.load()
    if saved_state:
        await queue_manager.restore_state(saved_state)
    await state_persistence.start_auto_save()

    logger.info("Audio playback service started")

    yield

    # Shutdown: save state and cleanup
    await state_persistence.stop_auto_save()
    await queue_manager.shutdown()
    logger.info("Audio playback service stopped")


app = FastAPI(
    title="Audio Playback Service",
    description="Audio playback service with multiple named queues. Play audio files through VLC with volume control, skip tracks, and manage multiple concurrent playback queues.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/queues", response_model=list[QueueInfo], operation_id="list_audio_queues")
async def list_audio_queues():
    """List all active audio playback queues.

    Returns information about each queue including name, volume level,
    whether it's currently playing, and how many audio files are queued.
    """
    queues = []
    for name in queue_manager.queue_names:
        queue = queue_manager.get_queue(name)
        if queue:
            queues.append(
                QueueInfo(
                    name=queue.name,
                    volume=queue.volume,
                    is_playing=queue.is_playing,
                    file_count=queue.file_count,
                )
            )
    return queues


@app.post("/queues/{name}", response_model=QueueStatus, operation_id="play_audio")
async def play_audio(name: str, request: QueueCreate):
    """Play audio files by adding them to a named playback queue.

    Creates a new audio queue if it doesn't exist, or appends files to an existing queue.
    Audio files start playing immediately in sequence. Use different queue names for
    concurrent playback (e.g., 'music', 'notifications', 'speech').

    Args:
        name: Unique identifier for the audio queue (e.g., 'music', 'tts', 'alerts')
        request: Audio files to play and optional volume setting
    """
    queue = await queue_manager.append_to_queue(
        name=name,
        files=request.files,
        volume=request.volume,
    )
    return QueueStatus(
        name=queue.name,
        volume=queue.volume,
        current_file=queue.current_file,
        current_position=queue.get_position(),
        current_duration=queue.get_duration(),
        remaining_files=queue.remaining_files,
        is_playing=queue.is_playing,
    )


@app.get("/queues/{name}", response_model=QueueStatus, operation_id="get_queue_status")
async def get_audio_queue_status(name: str):
    """Get the current status of an audio playback queue.

    Returns detailed information including current track, playback position,
    remaining files, volume level, and whether audio is currently playing.

    Args:
        name: Name of the audio queue to check
    """
    queue = queue_manager.get_queue(name)
    if not queue:
        raise HTTPException(status_code=404, detail=f"Queue '{name}' not found")

    return QueueStatus(
        name=queue.name,
        volume=queue.volume,
        current_file=queue.current_file,
        current_position=queue.get_position(),
        current_duration=queue.get_duration(),
        remaining_files=queue.remaining_files,
        is_playing=queue.is_playing,
    )


@app.delete("/queues", operation_id="stop_all_audio")
async def stop_all_audio_queues():
    """Stop all audio playback and remove all queues.

    Immediately stops any playing audio and removes all queues entirely.
    All queued files are discarded.
    """
    # Copy the list to avoid modification during iteration
    names = list(queue_manager.queue_names)
    for name in names:
        await queue_manager.remove_queue(name)
    if names:
        return {"message": f"Stopped queues: {', '.join(names)}"}
    return {"message": "No active queues to stop"}


@app.delete("/queues/{name}", operation_id="stop_audio")
async def stop_audio_queue(name: str):
    """Stop audio playback and remove a queue.

    Immediately stops any playing audio and removes the queue entirely.
    All queued files are discarded.

    Args:
        name: Name of the audio queue to stop and remove
    """
    success = await queue_manager.remove_queue(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Queue '{name}' not found")
    return {"message": f"Queue '{name}' removed"}


@app.put("/queues/{name}/volume", operation_id="set_volume")
async def set_audio_volume(name: str, request: VolumeUpdate):
    """Set the volume level for an audio playback queue.

    Changes the volume immediately, affecting currently playing audio.

    Args:
        name: Name of the audio queue
        request: New volume level (0-100)
    """
    success = await queue_manager.set_volume(name, request.volume)
    if not success:
        raise HTTPException(status_code=404, detail=f"Queue '{name}' not found")
    return {"message": f"Volume set to {request.volume}"}


@app.post("/queues/{name}/skip", operation_id="skip_track")
async def skip_audio_track(name: str):
    """Skip to the next audio file in the queue.

    Stops the currently playing audio and immediately starts the next file.
    If there are no more files, the queue becomes idle.

    Args:
        name: Name of the audio queue
    """
    success = await queue_manager.skip_track(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Queue '{name}' not found")
    return {"message": "Track skipped"}


@app.get("/health")
async def audio_service_health():
    """Check if the audio playback service is running.

    Returns service status and the number of active audio queues.
    """
    return {
        "status": "healthy",
        "active_queues": len(queue_manager.queue_names),
    }


if __name__ == "__main__":
    from fastmcp import FastMCP
    from fastmcp.server.openapi import RouteMap, MCPType, HTTPRoute, OpenAPITool

    def simplify_descriptions(
        route: HTTPRoute,
        component: OpenAPITool,
    ) -> None:
        """Strip verbose OpenAPI schema details, keep only the docstring."""
        if isinstance(component, OpenAPITool):
            # Keep only the first paragraph (the original docstring summary)
            component.description = component.description.split("\n\n")[0]

    # Convert to MCP server with customizations
    mcp = FastMCP.from_fastapi(
        app=app,
        route_maps=[
            # Exclude /health endpoint from MCP tools
            RouteMap(methods=["GET"], pattern=r"/health", mcp_type=MCPType.EXCLUDE),
        ],
        mcp_component_fn=simplify_descriptions,
    )

    base_url = os.getenv("MCP_URL_AUDIO_PLAYBACK")
    port = 8000
    if base_url is not None:
        parsed = urlparse(base_url)
        port = parsed.port

    mcp.run(
        transport="http",
        show_banner=True,
        port=port,
    )
