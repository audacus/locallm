import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_text_block, create_image_block
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
            content_blocks=[
                create_text_block(text="Convert following text to speech and play it: Hi there."),
                # create_image_block(url="graph.png"),
            ],
        ),
    ]

    context = {
        "voice": "af_heart",
    }
    GRAPH.invoke(
        input=Command(update={"messages": messages}),
        config=config,
        context=context,
        subgraphs=True
    )


if __name__ == "__main__":
    asyncio.run(main())
