# locallm

## Setup

```shell
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env
# add value for `LANGSMITH_API_KEY`.
```

## MCP server

```shell
.venv/bin/langgraph dev
# http://localhost:2024/mcp
```
