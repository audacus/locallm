from typing import TypedDict, Annotated, NotRequired

from langchain.agents.middleware.todo import Todo
from langchain.agents.middleware.types import OmitFromInput
from langgraph.graph import MessagesState


class OrchestratorState(MessagesState):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]


class OrchestratorOutputState(TypedDict):
    output: str
