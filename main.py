import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from graph.orchestrator import orchestrator_compiled

GRAPH = orchestrator_compiled


async def main():
    config: RunnableConfig = {
        # Ensure every run has a separate thread ID.
        "configurable": {"thread_id": str(uuid.uuid4())},
    }

    # Write graph diagram to file.
    image_bytes = GRAPH.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_bytes)

    # Simulate user input
    messages = [
        HumanMessage(
            content=""""
Convert following conversation to audio:
- Hey there!
- Hi!
- How are you?
- I'm fine thanks.
"""
        ),
    ]

    await GRAPH.ainvoke(
        input=Command(update={"messages": messages}),
        config=config,
        subgraphs=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
