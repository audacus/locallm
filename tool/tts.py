import base64
import fnmatch
import os
import time
from hashlib import md5
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from store.references import get_reference_key

load_dotenv()

openai_client = AsyncOpenAI(
    api_key="none",
    base_url=os.getenv("API_BASE_URL_MLX_AUDIO"),
)


class VoiceTextPart(BaseModel):
    voice: Literal["af_heart", "af_bella", "am_fenrir", "am_michael"] = Field(
        description="Voice for generating the speech.\nPrefix:\n- `af_` => American English female\n- `am_` => American English male"
    )
    text: str = Field(description="Text to convert to speech")


class TTSInput(BaseModel):
    voice_text_parts: list[VoiceTextPart] = Field(
        description="List of voice/text parts to convert to speech"
    )


class TTSGeneration(BaseModel):
    audio_file_path: str
    base64_data: str
    cached: bool
    input: str
    model: str
    voice: str


class TTSGenerationArtifact(BaseModel):
    audio_file_path_ref: str
    base64_data: str
    cached: bool
    input: str
    model: str
    voice: str


async def generate_speech(
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
        audio_file_path = generated_audio_dir.joinpath(
            f"{int(time.time())}-{generation_hash}.wav"
        )

        # Create directory or search for already generated audio.
        use_cached_file = False
        if not os.path.exists(generated_audio_dir):
            os.makedirs(generated_audio_dir)
        else:
            for file_name in os.listdir(generated_audio_dir):
                if fnmatch.fnmatch(file_name, f"*{generation_hash}.wav"):
                    audio_file_path = generated_audio_dir.joinpath(file_name)
                    print(
                        "Using cached file:",
                        voice_text_part.voice,
                        f'"{voice_text_part.text}"',
                    )
                    use_cached_file = True

        if not use_cached_file:
            # Effective TTS request.
            async with openai_client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice_text_part.voice,
                input=voice_text_part.text,
            ) as response:
                await response.stream_to_file(audio_file_path)

        try:
            with open(audio_file_path, "rb") as audio_file:
                base64_data = base64.b64encode(audio_file.read()).decode("utf-8")
        except Exception as e:
            raise IOError(f"There was an error reading the generated file: {e}")

        generations.append(
            TTSGeneration(
                base64_data=base64_data,
                cached=use_cached_file,
                audio_file_path=str(audio_file_path),
                input=voice_text_part.text,
                model=model,
                voice=voice_text_part.voice,
            )
        )

    return generations


@tool(
    "convert_text_to_speech",
    description="Converts a list of voice/text parts to speech and returns the references to the paths of the generated audio files.",
    args_schema=TTSInput,
)
async def convert_text_to_speech(
    voice_text_parts: list[VoiceTextPart],
    runtime: ToolRuntime,
) -> Command:
    if len(voice_text_parts) == 0:
        tool_error_message = ToolMessage(
            content="No parts given!",
            status="error",
            tool_call_id=runtime.tool_call_id,
        )
        tool_error_message.pretty_print()
        return Command(update={"messages": [tool_error_message]})

    try:
        generations = await generate_speech(voice_text_parts)
    except Exception as e:
        tool_error_message = ToolMessage(
            content=f"{e}",
            status="error",
            tool_call_id=runtime.tool_call_id,
        )
        tool_error_message.pretty_print()
        return Command(update={"messages": [tool_error_message]})

    message_lines = ["Successfully generated speech audio files:"]
    generation_artifacts: list[TTSGenerationArtifact] = []
    for generation in generations:
        text = (
            f"{generation.input[:32]}..."
            if len(generation.input) > 32
            else generation.input
        )

        reference_key_audio_file_path = await get_reference_key(
            runtime.store,
            runtime.config["configurable"]["context"]["user_id"],
            generation.audio_file_path,
        )
        message_lines.append(
            f"  - Input: {generation.voice}: {text} -> Voice audio saved to: {reference_key_audio_file_path}"
        )

        # Convert TTS generations to TTS generation artifacts.
        generation_artifacts.append(
            TTSGenerationArtifact(
                base64_data=generation.base64_data,
                cached=generation.cached,
                audio_file_path_ref=reference_key_audio_file_path,
                input=generation.input,
                model=generation.model,
                voice=generation.voice,
            )
        )

    tool_message = ToolMessage(
        content="\n".join(message_lines),
        tool_call_id=runtime.tool_call_id,
        artifact=generation_artifacts,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})
