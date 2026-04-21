import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field, SecretStr

load_dotenv()

general_model = ChatOpenAI(
    base_url=os.getenv("API_BASE_URL_MLX_LM"),
    api_key=SecretStr("none"),
    model=os.getenv("MODEL_GENERAL"),
    streaming=True,
    temperature=0,
    max_tokens=4096,
)


class TranslateInput(BaseModel):
    input_text: str = Field(description="The text to translate")
    input_language: str | None = Field(
        description="The language of the text to translate (if known)",
        default=None,
    )
    output_language: str = Field(
        description="The language into which the text is to be translated"
    )


class TranslationGeneration(BaseModel):
    input_text: str
    input_language: str
    output_text: str
    output_language: str


@tool(
    "translate_text",
    description="Translates text from one language to another.",
    args_schema=TranslateInput,
)
async def translate_text(
    input_text: str,
    input_language: str | None,
    output_language: str,
    runtime: ToolRuntime,
) -> Command:
    system_message = SystemMessage(
        content="\n".join(
            [
                "Translate the text in <input_text> to the language in <output_language>.",
                "Only output the translated text!",
            ]
        )
    )
    prompt_message_lines = []
    if input_language is not None:
        prompt_message_lines.append(
            f"<input_language>{input_language}</input_language>"
        )

    prompt_message_lines.extend(
        [
            f"<input_text>{input_text}</input_text>",
            f"<output_language>{output_language}</output_language>",
        ]
    )

    messages = [system_message, HumanMessage(content="\n".join(prompt_message_lines))]
    response: AIMessage = await general_model.ainvoke(input=messages)

    tool_message = ToolMessage(
        content=response.text.strip(),
        tool_call_id=runtime.tool_call_id,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})
