import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from graph import graph


async def main():
    compiled = graph.compile()
    config: RunnableConfig = {
        # Ensure every run has a separate thread ID.
        "configurable": {"thread_id": str(uuid.uuid4())},
    }

    # Write graph diagram to file.
    image_bytes = compiled.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(image_bytes)

    messages = [
        HumanMessage(
            content="What happens when an unstoppable force meets an immovable object?",
        ),
    ]
    compiled.invoke(input=Command(update={"messages": messages}), config=config)


if __name__ == "__main__":
    asyncio.run(main())
