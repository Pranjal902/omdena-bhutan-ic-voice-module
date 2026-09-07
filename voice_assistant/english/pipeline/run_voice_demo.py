"""Voice demo: mic -> Whisper -> Groq -> gTTS."""

from __future__ import annotations

import os

# PyTorch (Silero VAD) + NumPy/SciPy can each ship OpenMP; set before any of them load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

ENGLISH_ROOT = Path(__file__).resolve().parents[1]
VA_ROOT = ENGLISH_ROOT.parent
sys.path.insert(0, str(ENGLISH_ROOT))

import argparse

from pipeline.llm_groq import generate_reply
from pipeline.transcribe import transcribe_file
from pipeline.tts_output import speak_to_file
from pipeline.voice_capture import record_utterance


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice pipeline demo (VAD + Whisper + Groq + gTTS).")
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model for transcription.",
    )
    args = parser.parse_args()

    capture_dir = VA_ROOT / "outputs" / "english" / "pipeline_capture"
    audio_path = record_utterance(capture_dir)

    print("Transcribing with Whisper...")
    user_text = transcribe_file(str(audio_path), model_name=args.whisper_model)
    print("You said:", user_text)

    print("Calling Groq...")
    reply = generate_reply(user_text)
    print("Assistant:", reply)

    reply_audio = speak_to_file(
        reply,
        output_dir=VA_ROOT / "outputs" / "english" / "pipeline_replies",
    )
    print("Reply audio:", reply_audio)


if __name__ == "__main__":
    main()
