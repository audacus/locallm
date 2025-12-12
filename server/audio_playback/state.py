"""State persistence for the audio playback service."""

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path(__file__).parent / "state.json"


class StatePersistence:
    """Handles saving and loading queue state to disk."""

    def __init__(
        self,
        file_path: Path = DEFAULT_STATE_FILE,
        save_interval: float = 5.0,
    ):
        self.file_path = file_path
        self.save_interval = save_interval
        self._get_state_callback: Callable | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    def set_state_callback(self, callback: Callable):
        """Set callback to get current state."""
        self._get_state_callback = callback

    async def load(self) -> list[dict]:
        """Load state from file."""
        if not self.file_path.exists():
            logger.info("No state file found, starting fresh")
            return []

        try:
            content = self.file_path.read_text()
            data = json.loads(content)
            logger.info(f"Loaded state: {len(data.get('queues', []))} queues")
            return data.get("queues", [])
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load state: {e}")
            return []

    async def save(self, queues: list[dict]):
        """Save state to file."""
        try:
            data = {"queues": queues}
            self.file_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved state: {len(queues)} queues")
        except OSError as e:
            logger.error(f"Failed to save state: {e}")

    async def start_auto_save(self):
        """Start periodic auto-save task."""
        if self._task is not None:
            return

        self._running = True
        self._task = asyncio.create_task(self._auto_save_loop())
        logger.info(f"Started auto-save (interval: {self.save_interval}s)")

    async def stop_auto_save(self):
        """Stop auto-save and do final save."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Final save
        if self._get_state_callback:
            await self.save(self._get_state_callback())

        logger.info("Stopped auto-save")

    async def _auto_save_loop(self):
        """Periodic save loop."""
        while self._running:
            await asyncio.sleep(self.save_interval)

            if self._get_state_callback:
                await self.save(self._get_state_callback())
