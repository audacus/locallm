"""Pydantic models for the Audio Playback Service API."""

from pydantic import BaseModel, Field


class QueueCreate(BaseModel):
    """Audio files to play and playback settings."""

    files: list[str] = Field(
        default_factory=list,
        description="List of absolute file paths to audio files (e.g., MP3, WAV, FLAC) to play in sequence",
    )
    volume: int = Field(
        default=100,
        ge=0,
        le=100,
        description="Playback volume level from 0 (muted) to 100 (full volume)",
    )


class QueueStatus(BaseModel):
    """Current state of an audio playback queue."""

    name: str = Field(description="Unique name of this audio queue")
    volume: int = Field(description="Current volume level (0-100)")
    current_file: str | None = Field(description="Path to the audio file currently playing, or null if idle")
    current_position: float = Field(description="Current playback position in seconds")
    current_duration: float = Field(description="Total duration of the current track in seconds")
    remaining_files: list[str] = Field(description="Audio files waiting to be played after the current track")
    is_playing: bool = Field(description="True if audio is currently playing")


class QueueInfo(BaseModel):
    """Summary information about an audio playback queue."""

    name: str = Field(description="Unique name of this audio queue")
    volume: int = Field(description="Current volume level (0-100)")
    is_playing: bool = Field(description="True if audio is currently playing")
    file_count: int = Field(description="Total number of audio files in queue (including current)")


class VolumeUpdate(BaseModel):
    """New volume level for audio playback."""

    volume: int = Field(
        ge=0,
        le=100,
        description="Volume level from 0 (muted) to 100 (full volume)",
    )


class AppendFiles(BaseModel):
    """Additional audio files to add to an existing queue."""

    files: list[str] = Field(
        min_length=1,
        description="List of absolute file paths to audio files to append to the queue",
    )
