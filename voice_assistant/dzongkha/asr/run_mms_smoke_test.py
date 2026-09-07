"""Smoke test: load MMS (dzo), transcribe one clip from Jyoti-77/dzongkha-asr.

Quick check that model + dataset + transcription work. Not a pass-gate eval.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from datasets import Audio, load_dataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

VA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VA_ROOT))

import argparse

from dzongkha.asr.mms_asr import SAMPLE_RATE, load_mms_dzo, resample_if_needed, transcribe_array

DATASET_ID = "Jyoti-77/dzongkha-asr"


def _configure_stdout_utf8() -> None:
    """Windows consoles often use cp1252; Dzongkha needs UTF-8 for print."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def read_dataset_audio(audio_field: dict) -> np.ndarray:
    raw_bytes = audio_field.get("bytes")
    if raw_bytes:
        audio, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        return resample_if_needed(np.asarray(audio), int(sample_rate))

    path = audio_field.get("path")
    if path and Path(path).is_file():
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return resample_if_needed(np.asarray(audio), int(sample_rate))

    raise ValueError("Dataset audio row has no readable bytes or file path.")


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wav.write(str(path), SAMPLE_RATE, (clipped * 32767).astype(np.int16))


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Smoke test: one Dzongkha clip -> MMS (dzo) -> transcript."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index in Jyoti-77/dzongkha-asr train split (default: 0).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu).",
    )
    args = parser.parse_args()

    out_dir = VA_ROOT / "outputs" / "dzongkha" / "asr" / "smoke_test"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loading dataset {DATASET_ID} (row {args.index})...")
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))
    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"Index {args.index} out of range (dataset has {len(ds)} rows).")

    row = ds[args.index]
    reference = (row.get("text") or "").strip()
    audio = read_dataset_audio(row["audio"])
    duration_s = len(audio) / SAMPLE_RATE

    print(f"Reference text length: {len(reference)} chars")
    print(f"Audio duration: {duration_s:.2f}s @ {SAMPLE_RATE} Hz")

    print("\nLoading MMS (facebook/mms-1b-all) with dzo adapter...")
    print("(First run downloads ~1B weights — may take several minutes.)")
    load_start = time.perf_counter()
    model, processor, device = load_mms_dzo(device=args.device)
    load_seconds = time.perf_counter() - load_start
    print(f"Model ready on {device} ({load_seconds:.1f}s load time).")

    print("\nTranscribing...")
    prediction, proc_seconds = transcribe_array(audio, SAMPLE_RATE, model, processor, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"smoke_{stamp}_idx{args.index}.wav"
    save_wav(wav_path, audio)

    record = {
        "timestamp": stamp,
        "dataset": DATASET_ID,
        "row_index": args.index,
        "language_adapter": "dzo",
        "audio_path": str(wav_path),
        "audio_duration_s": round(duration_s, 3),
        "reference_text": reference,
        "prediction_text": prediction,
        "model_load_seconds": round(load_seconds, 3),
        "asr_process_seconds": round(proc_seconds, 3),
        "device": device,
        "smoke_test_passed": bool(prediction),
    }
    json_path = out_dir / f"smoke_{stamp}_idx{args.index}.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SMOKE TEST RESULT ===")
    print("Reference (from dataset):")
    print(reference or "(empty)")
    print("\nMMS prediction (dzo):")
    print(prediction or "(empty)")
    print("\n---")
    print(f"ASR time: {proc_seconds:.2f}s | RTF: {proc_seconds / duration_s:.3f}" if duration_s else f"ASR time: {proc_seconds:.2f}s")
    print(f"Saved audio: {wav_path}")
    print(f"Saved report: {json_path}")

    if prediction:
        print("\nSMOKE TEST: PASS (pipeline ran; got non-empty transcript)")
        print("Note: I cannot judge Dzongkha accuracy — share the JSON with a Bhutanese teammate.")
    else:
        print("\nSMOKE TEST: FAIL (empty transcript)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
