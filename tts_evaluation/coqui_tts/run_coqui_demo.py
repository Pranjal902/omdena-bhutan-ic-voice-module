"""Quick local Coqui TTS demo for English TTS evaluation."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import time

from coqui_engine import DEFAULT_MODEL, load_model, synthesize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one Coqui TTS WAV file from text.")
    parser.add_argument(
        "--text",
        default="Your registration is completed.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Coqui model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--out-dir",
        default="tts_evaluation/coqui_tts/outputs/coqui_demo",
        help="Directory where WAV will be saved.",
    )
    args = parser.parse_args()

    print("Loading Coqui model (first run may download files)...")
    load_started = time.perf_counter()
    load_seconds = load_model(model_name=args.model)
    print(f"Model ready in {load_seconds:.4f} sec (total wait: {time.perf_counter() - load_started:.4f} sec)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.out_dir) / f"coqui_demo_{timestamp}.wav"

    result = synthesize_text(
        text=args.text,
        output_path=output_file,
        model_name=args.model,
    )

    print("\n=== COQUI TTS DEMO RESULT ===")
    print(f"Status: {result.status}")
    print(f"Model: {result.model_name}")
    print(f"Output: {result.output_path}")
    print(f"Latency: {result.elapsed_seconds:.4f} sec")
    print(f"File Size: {result.file_size_bytes} bytes")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    main()
