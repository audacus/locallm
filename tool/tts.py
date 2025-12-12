import base64
import fnmatch
import os
import time
from hashlib import md5
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, ToolException
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

openai_client = OpenAI(
    api_key="none",
    base_url=os.getenv("API_BASE_URL_MLX_AUDIO"),
)


class VoiceTextPart(BaseModel):
    voice: Literal["af_heart", "am_michael"] = Field(
        description="Voice to generate the speech."
    )
    text: str = Field(description="Text to convert to speech.")


class TTSInput(BaseModel):
    voice_text_parts: list[VoiceTextPart] = Field(
        description="List of voice/text parts to convert to speech."
    )


class TTSGeneration(BaseModel):
    base64_data: str
    cached: bool
    file_path: str
    input: str
    model: str
    voice: str


def convert_text_to_speech(
    voice_text_parts: list[VoiceTextPart],
) -> list[TTSGeneration]:
    model = os.getenv("MODEL_TTS")

    generations: list[TTSGeneration] = []
    for voice_text_part in voice_text_parts:
        # Allow caching of generated audio.
        generation_hash = md5(
            f"{model}-{voice_text_part.voice}-{voice_text_part.text}".encode("utf-8")
        ).hexdigest()
        generated_audio_dir = Path(os.getenv("TTS_OUTPUT_DIR"))
        file_path = generated_audio_dir.joinpath(
            f"{int(time.time())}-{generation_hash}.wav"
        )

        # Create directory or search for already generated audio.
        use_cached_file = False
        if not os.path.exists(generated_audio_dir):
            os.makedirs(generated_audio_dir)
        else:
            for file_name in os.listdir(generated_audio_dir):
                if fnmatch.fnmatch(file_name, f"*{generation_hash}.wav"):
                    file_path = generated_audio_dir.joinpath(file_name)
                    use_cached_file = True

        if not use_cached_file:
            with openai_client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice_text_part.voice,
                input=voice_text_part.text,
            ) as response:
                response.stream_to_file(file_path)

        try:
            with open(file_path, "rb") as audio_file:
                base64_data = base64.b64encode(audio_file.read()).decode("utf-8")
        except FileNotFoundError as e:
            raise ToolException(f"There was an error reading the generated file: {e}")

        generations.append(
            TTSGeneration(
                base64_data=base64_data,
                cached=use_cached_file,
                file_path=str(file_path),
                input=voice_text_part.text,
                model=model,
                voice=voice_text_part.voice,
            )
        )

    return generations


@tool(
    "convert_texts_to_speech_audio_files",
    description="Converts a list of voice/text parts to speech and returns the paths to the generated audio files.",
    args_schema=TTSInput,
)
def call_tts(
    voice_text_parts: list[VoiceTextPart],
    runtime: ToolRuntime[None, MessagesState],
) -> Command:
    if len(voice_text_parts) == 0:
        raise ToolException("No parts given!")

    generations = convert_text_to_speech(voice_text_parts)

    message_lines = ["Successfully generated audio files:"]
    for generation in generations:
        text = (
            f"{generation.input[:32]}..."
            if len(generation.input) > 32
            else generation.input
        )
        message_lines.append(
            f"  - {generation.voice}: {text} -> {generation.file_path}"
        )

    tool_message = ToolMessage(
        content="\n".join(message_lines),
        tool_call_id=runtime.tool_call_id,
        artifact=generations,
    )

    tool_message.pretty_print()

    return Command(update={"messages": [tool_message]})
