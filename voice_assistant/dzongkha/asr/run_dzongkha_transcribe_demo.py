"""Mic -> Silero VAD -> MMS ASR (dzo) -> Dzongkha text (Part 1 demo)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

VA_ROOT = Path(__file__).resolve().parents[2]
ENGLISH_ROOT = VA_ROOT / "english"
sys.path.insert(0, str(VA_ROOT))
sys.path.insert(0, str(ENGLISH_ROOT))

import argparse

from pipeline.voice_capture import record_utterance

from dzongkha.asr.mms_asr import load_mms_dzo, transcribe_file


def _configure_stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _resolve_audio_path(args: argparse.Namespace, capture_dir: Path) -> Path:
    if args.audio:
        audio_path = Path(args.audio).expanduser().resolve()
        if not audio_path.is_file():
            raise SystemExit(f"Audio file not found: {audio_path}")
        capture_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        copied = capture_dir / f"from_file_{stamp}_{audio_path.name}"
        shutil.copy2(audio_path, copied)
        print(f"Using pre-recorded audio: {audio_path}")
        print(f"Copied to: {copied}")
        return copied

    return record_utterance(capture_dir)


def main() -> None:
    _configure_stdout_utf8()

    parser = argparse.ArgumentParser(
        description="Record Dzongkha speech and transcribe with MMS (dzo adapter)."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Skip mic; transcribe this WAV/MP3 path instead (for testing the demo path).",
    )
    args = parser.parse_args()

    capture_dir = VA_ROOT / "outputs" / "dzongkha" / "asr" / "mic_capture"
    transcript_dir = VA_ROOT / "outputs" / "dzongkha" / "asr" / "transcripts"

    print("Loading MMS (dzo)... first run downloads ~1B model weights.")
    load_start = time.perf_counter()
    model, processor, device = load_mms_dzo(device=args.device)
    load_seconds = time.perf_counter() - load_start
    print(f"Ready on {device} ({load_seconds:.1f}s load time).")

    if args.audio:
        print("\nFile mode — skipping microphone.")
    else:
        print("\nMic mode — speak after 'Speak now...', then pause briefly when done.")

    audio_path = _resolve_audio_path(args, capture_dir)

    print("\nTranscribing with MMS (Dzongkha / dzo)...")
    start = time.perf_counter()
    text, proc_seconds = transcribe_file(audio_path, model, processor, device)
    elapsed = time.perf_counter() - start

    print("\n--- Dzongkha transcript ---")
    print(text if text else "(empty)")
    print("-------------------------")
    print(f"ASR processing time: {proc_seconds:.2f}s (total wall: {elapsed:.2f}s)")

    transcript_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": stamp,
        "language": "dzo",
        "source": "file" if args.audio else "mic",
        "audio_path": str(audio_path),
        "transcript": text,
        "model_load_seconds": round(load_seconds, 3),
        "process_seconds": round(proc_seconds, 3),
        "device": device,
    }
    json_path = transcript_dir / f"transcript_{stamp}.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path = transcript_dir / f"transcript_{stamp}.txt"
    txt_path.write_text(text + "\n", encoding="utf-8")
    print("Saved:", json_path)
    print("Saved:", txt_path)

    if text:
        print("\nDemo complete. Share the transcript with a Dzongkha speaker to judge accuracy.")
    else:
        print("\nDemo finished but transcript is empty — check audio level or try again.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
