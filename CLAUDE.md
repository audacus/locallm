# Starting the app

## Servers

- mlx-lm: `.venv/bin/python server/mlx_lm/main.py`
- mlx-audio: `venv/bin/python server/mlx_audio/main.py`
- audio playback (MCP): `.venv/bin/python server/audio_playback/main.py`

## Agent

- App with hardcoded human input message: `.venv/bin/python main.py`
  - The agent needs all servers to run to function properly.
- Agent as MCP: `.venv/bin/langgraph dev`
