from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from graph.tools import get_tools


async def async_tool_node_wrapper(state: MessagesState) -> Command:
    """Wrap the pre-built `ToolNode` in an asynchronous node for passing the asynchronously initialized tool list."""
    tool_node = ToolNode(tools=await get_tools(), handle_tool_errors=False)

    result: ToolMessage | Command = await tool_node.ainvoke(input=state)

    if isinstance(result, ToolMessage):
        return Command(update={"messages": [result]})
    else:
        return result
