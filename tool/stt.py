import os

from mlx_whisper import transcribe

from dotenv import load_dotenv

load_dotenv()

#https://gitlab.inf.ethz.ch/ou-mtc-public/swiss-dial-samples
# file_path = "/Users/dbu/workspace/locallm/test_files/ch_zh_1682.wav"

file_path = "/Users/dbu/workspace/locallm/test_files/teams-call.mp3"

whisper_result = transcribe(
    audio=file_path,
    path_or_hf_repo=os.getenv("MODEL_STT"),
    verbose=True,
    word_timestamps=True,
)
with open("whisper_mlx_whisper.txt", "w") as f:
    f.write(whisper_result["text"])
