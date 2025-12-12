"""Audio queue management using python-vlc."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path

import vlc

logger = logging.getLogger(__name__)


class AudioQueue:
    """A named audio queue with its own VLC player instance."""

    def __init__(
        self,
        name: str,
        volume: int = 100,
        on_empty: Callable[["AudioQueue"], None] | None = None,
    ):
        self.name = name
        self._volume = volume
        self._on_empty = on_empty
        self._files: deque[str] = deque()
        self._current_file: str | None = None
        self._is_playing = False
        self._stop_requested = False

        # Create VLC instance and player
        self._vlc_instance = vlc.Instance("--no-xlib")
        self._player = self._vlc_instance.media_player_new()
        self._player.audio_set_volume(volume)

        # Set up event manager for end of media
        self._event_manager = self._player.event_manager()
        self._event_manager.event_attach(
            vlc.EventType.MediaPlayerEndReached,
            self._on_media_end,
        )
        self._event_manager.event_attach(
            vlc.EventType.MediaPlayerEncounteredError,
            self._on_media_error,
        )

        # Async event for signaling
        self._media_ended_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _on_media_end(self, event):
        """Callback when media finishes playing."""
        logger.debug(f"[{self.name}] Media ended: {self._current_file}")
        if self._loop:
            self._loop.call_soon_threadsafe(self._media_ended_event.set)

    def _on_media_error(self, event):
        """Callback when media encounters an error."""
        logger.error(f"[{self.name}] Media error: {self._current_file}")
        if self._loop:
            self._loop.call_soon_threadsafe(self._media_ended_event.set)

    @property
    def volume(self) -> int:
        return self._volume

    @volume.setter
    def volume(self, value: int):
        self._volume = max(0, min(100, value))
        self._player.audio_set_volume(self._volume)

    @property
    def current_file(self) -> str | None:
        return self._current_file

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def remaining_files(self) -> list[str]:
        return list(self._files)

    @property
    def file_count(self) -> int:
        count = len(self._files)
        if self._current_file:
            count += 1
        return count

    def get_position(self) -> float:
        """Get current playback position in seconds."""
        if not self._is_playing:
            return 0.0
        pos = self._player.get_time()
        return pos / 1000.0 if pos >= 0 else 0.0

    def get_duration(self) -> float:
        """Get current media duration in seconds."""
        if not self._current_file:
            return 0.0
        duration = self._player.get_length()
        return duration / 1000.0 if duration >= 0 else 0.0

    def append(self, files: list[str]):
        """Append files to the queue."""
        for file_path in files:
            if Path(file_path).exists():
                self._files.append(file_path)
                logger.info(f"[{self.name}] Added to queue: {file_path}")
            else:
                logger.warning(f"[{self.name}] File not found: {file_path}")

    def skip(self):
        """Skip the current track."""
        if self._is_playing:
            logger.info(f"[{self.name}] Skipping: {self._current_file}")
            self._player.stop()
            self._media_ended_event.set()

    def stop(self):
        """Stop the queue and clear all files."""
        self._stop_requested = True
        self._files.clear()
        self._player.stop()
        self._is_playing = False
        self._current_file = None
        self._media_ended_event.set()

    async def run(self):
        """Main loop that plays files from the queue."""
        self._loop = asyncio.get_running_loop()
        logger.info(f"[{self.name}] Queue started")

        while not self._stop_requested:
            if self._files:
                self._current_file = self._files.popleft()
                await self._play_file(self._current_file)
            else:
                # Queue is empty - check periodically or notify
                if self._on_empty:
                    self._on_empty(self)
                    break
                # Wait a bit before checking again
                await asyncio.sleep(0.5)

        self._is_playing = False
        self._current_file = None
        logger.info(f"[{self.name}] Queue stopped")

    async def _play_file(self, file_path: str):
        """Play a single file and wait for it to finish."""
        logger.info(f"[{self.name}] Playing: {file_path}")

        media = self._vlc_instance.media_new(file_path)
        self._player.set_media(media)
        self._player.audio_set_volume(self._volume)

        self._media_ended_event.clear()
        self._player.play()
        self._is_playing = True

        # Wait for media to end
        await self._media_ended_event.wait()

        self._is_playing = False

    def cleanup(self):
        """Release VLC resources."""
        self._player.stop()
        self._player.release()
        self._vlc_instance.release()

    def to_state(self) -> dict:
        """Export queue state for persistence."""
        files = list(self._files)
        if self._current_file:
            # Include current file at the start (it will be replayed)
            files.insert(0, self._current_file)
        return {
            "name": self.name,
            "volume": self._volume,
            "files": files,
        }

    @classmethod
    def from_state(
        cls,
        state: dict,
        on_empty: Callable[["AudioQueue"], None] | None = None,
    ) -> "AudioQueue":
        """Create a queue from saved state."""
        queue = cls(
            name=state["name"],
            volume=state.get("volume", 100),
            on_empty=on_empty,
        )
        if state.get("files"):
            queue.append(state["files"])
        return queue
