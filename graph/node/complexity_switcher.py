import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field, SecretStr

from graph.tools import get_tool_list
from prompt.complexity_switcher import COMPLEXITY_SWITCHER_PROMPT

complexity_switcher_model = ChatOpenAI(
    async_client=False,
    base_url=os.getenv("API_BASE_URL_MLX_LM"),
    api_key=SecretStr("none"),
    model=os.getenv("MODEL_COMPLEXITY_SWITCHER"),
    streaming=False,
    temperature=0,
)


async def complexity_switcher(state: MessagesState) -> Command:
    input_message = state["messages"][-1]

    class ComplexitySwitcherOutputSchema(BaseModel):
        is_complex: bool = Field(description="Whether the input request is complex.")

    # Prompt a separate LLM to determine if the input is complex or not.
    message = HumanMessage(
        content=COMPLEXITY_SWITCHER_PROMPT.format(
            input=input_message.text.strip(),
            tool_list=await get_tool_list(),
        )
    )
    response: ComplexitySwitcherOutputSchema = (
        complexity_switcher_model.with_structured_output(
            schema=ComplexitySwitcherOutputSchema,
        ).invoke(input=[message])
    )

    return Command(goto="enricher" if response.is_complex else "model")
