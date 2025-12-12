"""Queue manager for handling multiple audio queues."""

import asyncio
import logging

from audio_queue import AudioQueue

logger = logging.getLogger(__name__)


class QueueManager:
    """Manages multiple named audio queues."""

    def __init__(self):
        self._queues: dict[str, AudioQueue] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    @property
    def queue_names(self) -> list[str]:
        """Get all active queue names."""
        return list(self._queues.keys())

    def get_queue(self, name: str) -> AudioQueue | None:
        """Get a queue by name."""
        return self._queues.get(name)

    async def create_queue(
        self,
        name: str,
        files: list[str] | None = None,
        volume: int = 100,
    ) -> AudioQueue:
        """Create a new queue or get existing one."""
        async with self._lock:
            if name in self._queues:
                queue = self._queues[name]
                if files:
                    queue.append(files)
                return queue

            # Create new queue with auto-cleanup callback
            queue = AudioQueue(
                name=name,
                volume=volume,
                on_empty=self._on_queue_empty,
            )
            if files:
                queue.append(files)

            self._queues[name] = queue

            # Start the queue's run loop
            task = asyncio.create_task(queue.run())
            self._tasks[name] = task

            logger.info(f"Created queue: {name}")
            return queue

    def _on_queue_empty(self, queue: AudioQueue):
        """Callback when a queue becomes empty."""
        asyncio.create_task(self._cleanup_queue(queue.name))

    async def _cleanup_queue(self, name: str):
        """Remove an empty queue."""
        async with self._lock:
            if name in self._queues:
                queue = self._queues.pop(name)
                queue.cleanup()

                if name in self._tasks:
                    task = self._tasks.pop(name)
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                logger.info(f"Removed empty queue: {name}")

    async def append_to_queue(
        self,
        name: str,
        files: list[str],
        volume: int | None = None,
    ) -> AudioQueue:
        """Append files to an existing queue or create new one."""
        async with self._lock:
            if name in self._queues:
                queue = self._queues[name]
                queue.append(files)
                if volume is not None:
                    queue.volume = volume
                return queue

        # Queue doesn't exist, create it
        return await self.create_queue(name, files, volume or 100)

    async def remove_queue(self, name: str) -> bool:
        """Stop and remove a queue."""
        async with self._lock:
            if name not in self._queues:
                return False

            queue = self._queues.pop(name)
            queue.stop()
            queue.cleanup()

            if name in self._tasks:
                task = self._tasks.pop(name)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            logger.info(f"Removed queue: {name}")
            return True

    async def set_volume(self, name: str, volume: int) -> bool:
        """Set volume for a queue."""
        queue = self._queues.get(name)
        if queue:
            queue.volume = volume
            return True
        return False

    async def skip_track(self, name: str) -> bool:
        """Skip current track in a queue."""
        queue = self._queues.get(name)
        if queue:
            queue.skip()
            return True
        return False

    async def shutdown(self):
        """Stop all queues and cleanup."""
        async with self._lock:
            for name in list(self._queues.keys()):
                queue = self._queues.pop(name)
                queue.stop()
                queue.cleanup()

                if name in self._tasks:
                    task = self._tasks.pop(name)
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

            logger.info("All queues stopped")

    def get_state(self) -> list[dict]:
        """Export all queues state for persistence."""
        return [queue.to_state() for queue in self._queues.values()]

    async def restore_state(self, states: list[dict]):
        """Restore queues from saved state."""
        for state in states:
            await self.create_queue(
                name=state["name"],
                files=state.get("files", []),
                volume=state.get("volume", 100),
            )
        logger.info(f"Restored {len(states)} queues from state")
