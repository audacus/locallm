import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    word_timestamps=True,
    # initial_prompt="glossary",
    verbose=True,
    language="en",
)
