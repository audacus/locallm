import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.types import Command

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
                "user_id": "1",
            },
        }

        # Write graph diagram to file.
        image_bytes = graph_compiled.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(image_bytes)

        # Simulate user input
        messages = [
            HumanMessage(content=""""
Play following conversation as audio:
- How are you?
- I'm fine thanks.
"""),
        ]

        await graph_compiled.ainvoke(
            input=Command(update={"messages": messages}),
            config=config,
            subgraphs=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
