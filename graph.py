from typing import TypedDict, Annotated

from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.types import Command

orchestrator_llm = MLXPipeline.from_model_id(
    model_id="mlx-community/Orchestrator-8B-4bit",
    pipeline_kwargs={
        "temp": 0.0,
        "max_tokens": 4096,
        "verbose": True,
    },
)

orchestrator_chat_model = ChatMLX(
    llm=orchestrator_llm,
    disable_streaming=True,
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    pass


def orchestrate(state: State) -> Command:
    response = orchestrator_chat_model.invoke(input=state["messages"])
    return Command(
        update={
            "messages": [response],
        }
    )


graph = StateGraph(state_schema=State)
graph.add_node("orchestrate", orchestrate)
graph.set_entry_point("orchestrate")
graph.set_finish_point("orchestrate")
