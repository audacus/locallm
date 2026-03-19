import asyncio
import base64
import mimetypes
import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages.content import create_text_block, create_file_block
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.types import Command

from graph.models import OrchestratorContext
from graph.orchestrator import orchestrator_graph

load_dotenv()

GRAPH = orchestrator_graph
DB_URI = os.environ.get("POSTGRES_DB_URI")


async def main():
    async with (
        AsyncPostgresStore.from_conn_string(DB_URI) as store,
        AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
    ):
        await store.setup()
        await checkpointer.setup()

        graph_compiled = GRAPH.compile(
            checkpointer=checkpointer,
            store=store,
        )

        config: RunnableConfig = {
            # Ensure every run has a separate thread ID.
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            },
        }

        # Ensure every run has a separate thread ID.
        context = OrchestratorContext(user_id="1")

        # Write graph diagram to file.
        image_bytes = graph_compiled.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(image_bytes)

        # Create file attachment.
        file_path = "/Users/dbu/workspace/locallm/test_files/ch-8s.wav"
        mime_type, encoding = mimetypes.guess_type(file_path)
        with open(file_path, "rb") as f:
            file_content = f.read()
            base64_data = base64.b64encode(file_content).decode("ascii")

        # Simulate user input
        messages = [

            # HumanMessage(content=""""
            # Play following conversation as audio:
            # - Hi there. How are you?
            # - Hey. I'm fine thanks, and you?
            # - Me too, thanks.
            # """),

            # HumanMessage(
            #     content_blocks=[
            #         create_text_block(
            #             text="Play the attached file as background music and list all queues. Wait for 10 seconds. After that stop the music and list all queues again.",
            #         ),
            #         create_file_block(base64=base64_data, mime_type=mime_type),
            #     ],
            # ),

            HumanMessage(
                content_blocks=[
                    create_text_block(
                        # text="Transcribe given audio, translate the text to English and play it as speech audio.",
                        text="Translate the given audio to English.",
                    ),
                    create_file_block(base64=base64_data, mime_type=mime_type),
                ],
            ),

        ]

        await graph_compiled.ainvoke(
            input=Command(update={"messages": messages}),
            config=config,
            context=context,
            subgraphs=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
