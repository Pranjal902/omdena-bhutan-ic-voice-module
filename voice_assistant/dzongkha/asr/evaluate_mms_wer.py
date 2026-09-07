"""Dzongkha MMS ASR baseline: Jyoti-77/dzongkha-asr, WER/CER, clean + 15 dB SNR."""

from __future__ import annotations

import sys
from pathlib import Path

VA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VA_ROOT))

import argparse
import csv
import io
import os
import statistics
import time
from datetime import datetime

import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from datasets import Audio, load_dataset
from jiwer import cer, wer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from shared.noise_utils import add_white_noise_snr

from dzongkha.asr.mms_asr import (
    SAMPLE_RATE,
    load_mms_dzo,
    normalize_dzo_text,
    resample_if_needed,
    tokenize_for_wer,
    transcribe_array,
)

PASS_GATE_SNR = 15.0


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wav.write(str(path), sample_rate, (clipped * 32767).astype(np.int16))


def load_dzongkha_dataset(limit: int | None = None):
    ds = load_dataset("Jyoti-77/dzongkha-asr", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def read_dataset_audio(audio_field: dict) -> tuple[np.ndarray, int]:
    raw_bytes = audio_field.get("bytes")
    if raw_bytes:
        audio, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        return resample_if_needed(np.asarray(audio), int(sample_rate)), SAMPLE_RATE

    path = audio_field.get("path")
    if path and Path(path).is_file():
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return resample_if_needed(np.asarray(audio), int(sample_rate)), SAMPLE_RATE

    raise ValueError("Dataset audio row has no readable bytes or file path.")


def evaluate(
    limit: int | None,
    output_dir: Path,
    report_path: Path,
    snr_levels: list[float | None],
    noise_seed: int,
    device: str | None,
) -> None:
    np.random.seed(noise_seed)
    dataset = load_dzongkha_dataset(limit=limit)

    print("Loading MMS model (facebook/mms-1b-all) with dzo adapter...")
    load_start = time.perf_counter()
    model, processor, device = load_mms_dzo(device=device)
    print(f"Model ready on {device} ({time.perf_counter() - load_start:.1f}s load time, one-time).")

    rows: list[dict] = []

    for idx, item in enumerate(dataset):
        utterance_id = f"D{idx + 1:04d}"
        reference = normalize_dzo_text(item["text"])
        clean_audio, _ = read_dataset_audio(item["audio"])
        audio_seconds = len(clean_audio) / SAMPLE_RATE

        for snr in snr_levels:
            if snr is None:
                label = "clean"
                test_audio = clean_audio
            else:
                label = f"snr_{int(snr)}db"
                test_audio = add_white_noise_snr(clean_audio, snr)

            test_wav = output_dir / label / f"{utterance_id}.wav"
            save_wav(test_wav, test_audio)

            prediction, proc_seconds = transcribe_array(
                test_audio, SAMPLE_RATE, model, processor, device
            )
            prediction_norm = normalize_dzo_text(prediction)
            ref_wer = tokenize_for_wer(reference)
            pred_wer = tokenize_for_wer(prediction_norm)
            wer_score = wer(ref_wer, pred_wer)
            cer_score = cer(reference, prediction_norm)
            rtf = proc_seconds / audio_seconds if audio_seconds > 0 else 0.0

            row = {
                "id": utterance_id,
                "ground_truth": reference,
                "noise_condition": label,
                "snr_db": "" if snr is None else snr,
                "prediction": prediction_norm,
                "wer": round(wer_score, 4),
                "cer": round(cer_score, 4),
                "audio_seconds": round(audio_seconds, 3),
                "process_seconds": round(proc_seconds, 3),
                "rtf": round(rtf, 4),
            }
            rows.append(row)
            print(
                f"[{utterance_id} | {label}] wer={row['wer']:.4f} cer={row['cer']:.4f} "
                f"rtf={row['rtf']:.3f}"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pass_gate_label = f"snr_{int(PASS_GATE_SNR)}db"
    gate_rows = [r for r in rows if r["noise_condition"] == pass_gate_label]
    clean_rows = [r for r in rows if r["noise_condition"] == "clean"]

    mean_gate_wer = statistics.mean(r["wer"] for r in gate_rows) if gate_rows else None
    mean_gate_cer = statistics.mean(r["cer"] for r in gate_rows) if gate_rows else None
    mean_clean_wer = statistics.mean(r["wer"] for r in clean_rows) if clean_rows else None
    mean_clean_cer = statistics.mean(r["cer"] for r in clean_rows) if clean_rows else None
    mean_rtf = statistics.mean(r["rtf"] for r in rows)

    snr_labels = sorted({r["noise_condition"] for r in rows if r["noise_condition"] != "clean"})

    print("\n=== DZONGKHA MMS ASR BASELINE ===")
    print("Language adapter: dzo (Dzongkha, not Tibetan bod)")
    print(f"Utterances: {len(dataset)}")
    print(f"Total runs: {len(rows)}")
    if mean_clean_wer is not None:
        print(f"Mean WER (clean): {mean_clean_wer:.4f}")
        print(f"Mean CER (clean): {mean_clean_cer:.4f}")

    for label in snr_labels:
        subset = [r for r in rows if r["noise_condition"] == label]
        if subset:
            print(
                f"Mean WER ({label}): {statistics.mean(r['wer'] for r in subset):.4f} | "
                f"CER: {statistics.mean(r['cer'] for r in subset):.4f}"
            )

    print(f"Mean RTF (all runs): {mean_rtf:.4f}")

    if mean_gate_wer is not None:
        passed_wer = mean_gate_wer < 0.20
        passed_cer = mean_gate_cer is not None and mean_gate_cer < 0.20
        print(f"Mean WER (balanced noise level {int(PASS_GATE_SNR)} dB SNR): {mean_gate_wer:.4f}")
        print(f"Mean CER (balanced noise level {int(PASS_GATE_SNR)} dB SNR): {mean_gate_cer:.4f}")
        print(
            f"Pass/fail WER (< 0.20 at {int(PASS_GATE_SNR)} dB SNR): "
            f"{'PASS' if passed_wer else 'FAIL'}"
        )
        print(
            f"Pass/fail CER (< 0.20 at {int(PASS_GATE_SNR)} dB SNR): "
            f"{'PASS' if passed_cer else 'FAIL'}"
        )
        worst = sorted(gate_rows, key=lambda r: r["wer"], reverse=True)[:5]
        if worst:
            print(f"\nWorst WER at {int(PASS_GATE_SNR)} dB SNR (review these first):")
            for row in worst:
                print(f"  {row['id']} wer={row['wer']:.4f} cer={row['cer']:.4f}")
    else:
        print(
            f"\nNote: pass/fail uses {int(PASS_GATE_SNR)} dB SNR (balanced noise level) "
            "— not included in this run (exploratory SNR only)."
        )

    print("Note: WER uses tsheg/space tokenization for Dzongkha; track CER too.")
    print(f"\nReport: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dzongkha MMS ASR (WER/CER).")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit utterances (0 = full dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(VA_ROOT / "outputs" / "dzongkha" / "asr" / "eval_audio"),
        help="Folder for generated WAV samples.",
    )
    parser.add_argument(
        "--report-csv",
        default=str(VA_ROOT / "outputs" / "dzongkha" / "asr" / "dzongkha_mms_wer_results.csv"),
        help="CSV report output path (timestamp appended).",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Also evaluate clean audio (reference only — not used for pass/fail).",
    )
    parser.add_argument(
        "--snr-levels",
        type=float,
        nargs="+",
        help="Custom SNR levels in dB (default: 15 only). Include 15 for pass/fail summary.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Random seed for noise generation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu).",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    report_path = Path(args.report_csv).with_name(
        Path(args.report_csv).stem + f"_{timestamp}.csv"
    )

    if args.snr_levels:
        snr_levels = list(args.snr_levels)
    else:
        snr_levels = [PASS_GATE_SNR]
    if args.include_clean:
        snr_levels = [None] + snr_levels

    evaluate(
        limit=limit,
        output_dir=output_dir,
        report_path=report_path,
        snr_levels=snr_levels,
        noise_seed=args.noise_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
