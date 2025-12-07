import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from graph.orchestrator import orchestrator_compiled

# from runnable.tts import tts_compiled

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
            content="""Play following texts as audio:
- Hello there!
- How are you?"""
        ),
    ]

    context = {
        "voice": "af_heart",
    }
    GRAPH.invoke(
        input=Command(update={"messages": messages}),
        config=config,
        context=context,
        subgraphs=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
