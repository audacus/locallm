import fnmatch
import os
import time
from hashlib import md5
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from graph.models import OrchestratorContext
from store.references import get_reference_value, get_reference_key

load_dotenv()

openai_client = AsyncOpenAI(
    api_key="none",
    base_url=os.getenv("API_BASE_URL_MLX_AUDIO"),
)


class TranscriptionINput(BaseModel):
    file_path_refs: list[str] = Field(
        description="List of reference keys (e.g., 'REF_1', 'REF_2') pointing to audio file paths"
    )


class TranscriptionFileObject(BaseModel):
    text: str
    language: str


class TranscriptionGeneration(BaseModel):
    audio_file_path: str
    cached: bool
    language: str
    model: str
    text_content: str
    text_file_path: str


class TranscriptionGenerationArtifact(BaseModel):
    audio_file_path_ref: str
    cached: bool
    language: str
    model: str
    text_content: str
    text_file_path_ref: str


async def transcribe_audio(
    audio_file_paths: list[str],
) -> list[TranscriptionGeneration]:
    model = os.getenv("MODEL_STT")

    generations: list[TranscriptionGeneration] = []
    for audio_file_path in audio_file_paths:
        # Allow caching of created transcription files.
        try:
            with open(audio_file_path, "rb") as audio_file:
                audio_file_content = audio_file.read()
        except Exception as e:
            raise IOError(f"There was an error reading the input audio file: {e}")

        # Create hash of audio input file content.
        audio_file_content_hash = md5(audio_file_content).hexdigest()

        generation_hash = md5(
            f"{model}-{audio_file_content_hash}".encode("utf-8")
        ).hexdigest()
        transcription_output_dir = Path(os.getenv("STT_OUTPUT_DIR")).joinpath(
            "transcription"
        )
        transcription_output_dir.mkdir(parents=True, exist_ok=True)
        transcription_file_path = transcription_output_dir.joinpath(
            f"{int(time.time())}-{generation_hash}.json"
        )

        # Create directory or search for already created transcription file.
        use_cached_file = False
        if not os.path.exists(transcription_output_dir):
            os.makedirs(transcription_output_dir)
        else:
            for file_name in os.listdir(transcription_output_dir):
                if fnmatch.fnmatch(file_name, f"*{generation_hash}.json"):
                    transcription_file_path = transcription_output_dir.joinpath(
                        file_name
                    )
                    print("Using cached file...")
                    use_cached_file = True
                    try:
                        with open(transcription_file_path, "r") as transcription_file:
                            transcription_file_object = (
                                TranscriptionFileObject.model_validate_json(
                                    transcription_file.read()
                                )
                            )
                    except Exception as e:
                        raise IOError(
                            f"There was an error reading the transcription file: {e}"
                        )

        if not use_cached_file:
            # Transcription request.
            async with openai_client.audio.transcriptions.with_streaming_response.create(
                model=model,
                file=audio_file_content,
            ) as response:
                response_json = await response.json()
                transcription_file_object = TranscriptionFileObject(
                    text=response_json.get("text", "").strip(),
                    language=response_json.get("language", ""),
                )
                try:
                    with transcription_file_path.open("w") as transcription_file:
                        transcription_file.write(
                            transcription_file_object.model_dump_json()
                        )
                except Exception as e:
                    raise IOError(
                        f"There was an error writing the transcription file: {e}"
                    )

        generations.append(
            TranscriptionGeneration(
                audio_file_path=audio_file_path,
                cached=use_cached_file,
                language=transcription_file_object.language,
                model=model,
                text_content=transcription_file_object.text,
                text_file_path=str(Path(transcription_file_path)),
            )
        )

    return generations


@tool(
    "transcribe_audio",
    description="Transcribes a list of speech/audio files and returns a list of created JSON files containing the transcribed text and the detected language.",
    args_schema=TranscriptionINput,
)
async def call_transcribe_audio(
    file_path_refs: list[str],
    runtime: ToolRuntime[OrchestratorContext, MessagesState],
) -> Command:
    if len(file_path_refs) == 0:
        tool_error_message = ToolMessage(
            content="No files given!",
            status="error",
            tool_call_id=runtime.tool_call_id,
        )
        tool_error_message.pretty_print()
        return Command(update={"messages": [tool_error_message]})

    # Resolve reference keys to actual file paths.
    file_paths: list[str] = []
    for ref in file_path_refs:
        file_path = await get_reference_value(
            runtime.store, runtime.context["user_id"], ref
        )
        if file_path is None:
            tool_error_message = ToolMessage(
                content=f"Reference '{ref}' not found.",
                status="error",
                tool_call_id=runtime.tool_call_id,
            )
            tool_error_message.pretty_print()
            return Command(update={"messages": [tool_error_message]})
        file_paths.append(file_path)

    try:
        generations = await transcribe_audio(file_paths)
    except Exception as e:
        tool_error_message = ToolMessage(
            content=f"{e}",
            status="error",
            tool_call_id=runtime.tool_call_id,
        )
        tool_error_message.pretty_print()
        return Command(update={"messages": [tool_error_message]})

    message_lines = ["Successfully created JSON files containing the transcribed text:"]
    generation_artifacts: list[TranscriptionGenerationArtifact] = []
    for generation in generations:
        text = (
            f"{generation.text_content[:32].strip()}... (truncated)"
            if len(generation.text_content) > 32
            else generation.text_content
        )

        reference_key_audio_file_path = await get_reference_key(
            runtime.store,
            runtime.context["user_id"],
            generation.audio_file_path,
        )
        reference_key_text_file_path = await get_reference_key(
            runtime.store,
            runtime.context["user_id"],
            generation.text_file_path,
        )
        message_lines.append(
            f"  - Input audio file: {reference_key_audio_file_path} -> Transcribed text saved to: {reference_key_text_file_path}"
        )

        # Convert transcription generations to transcription generation artifacts.
        generation_artifacts.append(
            TranscriptionGenerationArtifact(
                audio_file_path_ref=reference_key_audio_file_path,
                cached=generation.cached,
                language=generation.language,
                model=generation.model,
                text_content=text,
                text_file_path_ref=reference_key_text_file_path,
            )
        )

    tool_message = ToolMessage(
        content="\n".join(message_lines),
        tool_call_id=runtime.tool_call_id,
        artifact=generation_artifacts,
    )
    tool_message.pretty_print()
    return Command(update={"messages": [tool_message]})
