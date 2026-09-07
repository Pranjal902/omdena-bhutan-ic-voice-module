"""Text demo: user text -> Groq -> gTTS audio."""

from __future__ import annotations

import sys
from pathlib import Path

ENGLISH_ROOT = Path(__file__).resolve().parents[1]
VA_ROOT = ENGLISH_ROOT.parent
sys.path.insert(0, str(ENGLISH_ROOT))

import argparse

from pipeline.llm_groq import generate_reply
from pipeline.tts_output import speak_to_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Text pipeline demo (Groq + gTTS).")
    parser.add_argument(
        "--text",
        default="Where is permit office in Thimphu?",
        help="User question text.",
    )
    args = parser.parse_args()

    print("User:", args.text)
    print("Calling Groq...")
    reply = generate_reply(args.text)
    print("Assistant:", reply)

    audio_path = speak_to_file(
        reply,
        output_dir=VA_ROOT / "outputs" / "english" / "pipeline_replies",
    )
    print("Reply audio:", audio_path)


if __name__ == "__main__":
    main()
