import base64
import fnmatch
import os
import time
from hashlib import md5
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.runtime import Runtime
from langgraph.types import Command
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

openai_client = OpenAI(
    api_key="none",
    base_url=os.getenv("API_BASE_URL_TTS"),
)


class TTSGeneration(BaseModel):
    base64_data: str
    cached: bool
    file_path: str
    input: str
    model: str
    voice: str


class TTSState(MessagesState):
    pass


class TTSOutputState(MessagesState):
    generations: list[TTSGeneration]


class TTSContext(TypedDict):
    voice: str


def text_to_speech(
    state: TTSState,
    runtime: Runtime[TTSContext],
) -> Command:
    model = os.getenv("MODEL_TTS")
    # Ensure a voice is set.
    voice = (
        runtime.context["voice"]
        if runtime.context and "voice" in runtime.context
        else "am_michael"
    )

    texts: list[str] = state["messages"][-1].content
    generations: list[TTSGeneration] = []
    for text in texts:
        # Allow caching of generated audio.
        generation_hash = md5(f"{model}-{voice}-{text}".encode("utf-8")).hexdigest()
        generated_audio_dir = "generated_audio"
        file_path = f"{generated_audio_dir}/{int(time.time())}-{generation_hash}.wav"

        # Create directory or search for already generated audio.
        use_cached_file = False
        if not os.path.exists(generated_audio_dir):
            os.makedirs(generated_audio_dir)
        else:
            for file_name in os.listdir(generated_audio_dir):
                if fnmatch.fnmatch(file_name, f"*{generation_hash}.wav"):
                    file_path = f"{generated_audio_dir}/{file_name}"
                    use_cached_file = True

        if not use_cached_file:
            with openai_client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
            ) as response:
                response.stream_to_file(file_path)

        with open(file_path, "rb") as audio_file:
            base64_data = base64.b64encode(audio_file.read()).decode("utf-8")

        generations.append(
            TTSGeneration(
                base64_data=base64_data,
                cached=use_cached_file,
                file_path=file_path,
                input=text,
                model=model,
                voice=voice,
            )
        )

    message_lines = ["Successfully generated audio files:"]
    for generation in generations:
        message_lines.append(f"  - {generation.file_path}")

    return Command(
        update={
            "messages": [AIMessage(content="\n".join(message_lines))],
            "generations": generations,
        },
    )


tts_graph = StateGraph(
    state_schema=TTSState,
    context_schema=TTSContext,
    output_schema=TTSOutputState,
)
tts_graph.add_node("text_to_speech", text_to_speech)
tts_graph.set_entry_point("text_to_speech")
tts_graph.set_finish_point("text_to_speech")

tts_compiled = tts_graph.compile()
