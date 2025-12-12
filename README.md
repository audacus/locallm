# locallm

## Setup

```shell
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env
# Add value for `LANGSMITH_API_KEY`.
```

## Servers

```shell
# mlx-lm
.venv/bin/python server/mlx_lm/main.py

# mlx-audio
.venv/bin/python server/mlx_audio/main.py

# audio playback (MCP)
.venv/bin/python server/audio_playback/main.py
```

## LangGraph API server

```shell
.venv/bin/langgraph dev
# MCP: http://localhost:2024/mcp
```
