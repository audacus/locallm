import os

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from tool.tts import call_tts

load_dotenv()

mcp_client = MultiServerMCPClient(
    {
        "audio_playback": {
            "transport": "http",
            "url": os.getenv("MCP_URL_AUDIO_PLAYBACK"),
        }
    }
)


async def get_tools() -> list[BaseTool]:
    # Use get_tools() instead of manually managing sessions.
    # MultiServerMCPClient is stateless by default - each tool invocation
    # creates a fresh session, executes the tool, and cleans up.
    mcp_tools = await mcp_client.get_tools()

    return [
        call_tts,
        *mcp_tools,
    ]


async def get_tool_list() -> str:
    tools = await get_tools()
    return "\n".join(
        [f"- {tool.name}: {tool.description.split("\n")[0]}" for tool in tools]
    )
