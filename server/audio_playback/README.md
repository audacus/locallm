# Audio Playback Service

Multi-queue audio playback service with REST API and MCP integration. Each queue plays files serially with independent volume control.

## Setup

```bash
# macOS
brew install vlc

pip install -r requirements.txt

python main.py
# runs on http://localhost:8000
```

API docs: `http://localhost:8000/docs`

## MCP Integration

This service can be exposed as an MCP server using FastMCP. The following tools are available:

| MCP Tool                 | Description                                           |
|--------------------------|-------------------------------------------------------|
| `play_audio`             | Play audio files by adding them to a named queue      |
| `list_audio_queues`      | List all active audio playback queues                 |
| `get_audio_queue_status` | Get current status of an audio playback queue         |
| `stop_audio_queue`       | Stop playback and remove a queue                      |
| `set_audio_volume`       | Set volume level for a queue (0-100)                  |
| `skip_audio_track`       | Skip to the next audio file in the queue              |
| `append_audio_files`     | Add more audio files to an existing queue             |
| `audio_service_health`   | Check if the audio playback service is running        |

## REST API Endpoints

| Endpoint                | Method | Description                                          |
|-------------------------|--------|------------------------------------------------------|
| `/queues`               | GET    | List all active queues (with name, volume, status)   |
| `/queues/{name}`        | POST   | Create queue or append files                         |
| `/queues/{name}`        | GET    | Get queue status (current file, position, remaining) |
| `/queues/{name}`        | DELETE | Stop and remove queue                                |
| `/queues/{name}/volume` | PUT    | Set volume (0-100)                                   |
| `/queues/{name}/skip`   | POST   | Skip current track                                   |
| `/queues/{name}/append` | POST   | Append files to existing queue                       |
| `/health`               | GET    | Health check                                         |

## Example Usage

```bash
# Create a background music queue at 30% volume
curl -X POST "http://localhost:8000/queues/background-music" \
  -H "Content-Type: application/json" \
  -d '{"files": ["/path/to/song1.mp3", "/path/to/song2.mp3"], "volume": 30}'

# Create a conversation queue at full volume
curl -X POST "http://localhost:8000/queues/conversation" \
  -H "Content-Type: application/json" \
  -d '{"files": ["/path/to/tts1.wav"], "volume": 100}'

# List all queues
curl "http://localhost:8000/queues"
```

## Notes

- Queues auto-close when empty
- State persists across restarts (saved to `state.json`)
- Supports all VLC-compatible formats (MP3, WAV, OGG, FLAC, etc.)

## Architecture

 ```
 ┌─────────────────────────────────────────────────────┐
 │          Audio Playback Service (Background)        │
 │                                                     │
 │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
 │  │  Queue 1    │  │  Queue 2    │  │  Queue N    │  │
 │  │ (bg music)  │  │ (convo)     │  │ (alerts)    │  │
 │  │  VLC Player │  │  VLC Player │  │  VLC Player │  │
 │  └─────────────┘  └─────────────┘  └─────────────┘  │
 │                                                     │
 │  ┌───────────────────────────────────────────────┐  │
 │  │             Queue Manager (async)             │  │
 │  └───────────────────────────────────────────────┘  │
 │                                                     │
 │  ┌───────────────────────────────────────────────┐  │
 │  │        FastAPI REST / FastMCP Layer           │  │
 │  └───────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────┘
 ```
