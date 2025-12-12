import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import SecretStr

from graph.tools import get_tool_list
from graph.utils import strip_thinking
from prompt.input_enricher import INPUT_ENRICHER_PROMPT

input_enricher_model = ChatOpenAI(
    async_client=False,
    base_url=os.getenv("API_BASE_URL_MLX_LM"),
    api_key=SecretStr("none"),
    model=os.getenv("MODEL_INPUT_ENRICHER"),
    streaming=False,
    temperature=0,
    max_tokens=4096,
)


async def input_enricher(state: MessagesState) -> Command:
    input_message = state["messages"][-1]

    input_message.pretty_print()

    # Prompt an LLM separately to enrich the original input.
    message = HumanMessage(
        content=INPUT_ENRICHER_PROMPT.format(
            input=input_message.text.strip(),
            tool_list=await get_tool_list(),
        )
    )
    response: AIMessage = input_enricher_model.invoke(input=[message])

    enriched_input_message = HumanMessage(content=strip_thinking(response.content))

    enriched_input_message.pretty_print()

    # Prepend the enriched input prompt.
    # Thought:
    # Prepare the model with the enriched prompt as a broader idea on how to solve the issue.
    # Prepend it, because it might include invented instruction that do not align with the original input (distortion through the enrichment).
    # And so the original input comes after "getting an idea" of how to process "such an input" (~two-shot training).
    return Command(update={"messages": [enriched_input_message, input_message]})
