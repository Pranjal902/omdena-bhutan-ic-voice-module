"""Quick local gTTS demo for English TTS evaluation."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse

from gtts_engine import synthesize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one gTTS MP3 file from text.")
    parser.add_argument(
        "--text",
        default="Your registration is completed.",
        help="Text to synthesize.",
    )
    parser.add_argument("--lang", default="en", help="Language code (default: en).")
    parser.add_argument("--tld", default="com", help="Google TLD voice domain (default: com).")
    parser.add_argument(
        "--out-dir",
        default="tts_evaluation/gtts/outputs/gtts_demo",
        help="Directory where MP3 will be saved.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.out_dir) / f"gtts_demo_{timestamp}.mp3"

    result = synthesize_text(
        text=args.text,
        output_path=output_file,
        language=args.lang,
        tld=args.tld,
    )

    print("=== gTTS DEMO RESULT ===")
    print(f"Status: {result.status}")
    print(f"Output: {result.output_path}")
    print(f"Latency: {result.elapsed_seconds:.4f} sec")
    print(f"File Size: {result.file_size_bytes} bytes")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    main()
