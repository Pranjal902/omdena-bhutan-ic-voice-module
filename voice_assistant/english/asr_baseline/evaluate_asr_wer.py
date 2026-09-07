"""English ASR baseline: synthesize speech, add noise, Whisper, WER."""

from __future__ import annotations

import sys
from pathlib import Path

VA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VA_ROOT))

from datetime import datetime
import argparse
import csv
import os
import re
import shutil
import statistics
from tempfile import gettempdir

import numpy as np
import scipy.io.wavfile as wav
import whisper
from gtts import gTTS
from jiwer import wer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from shared.data.load_utterances import DEFAULT_CSV, load_test_utterances
from shared.noise_utils import add_white_noise_snr

# Pass gate: WER < 20% on noisy audio at 15 dB SNR (see docs/JOURNEY.md).
PASS_GATE_SNR = 15.0
EXPLORATORY_SNR_LEVELS: list[float] = [20.0, 15.0, 10.0]


def ensure_ffmpeg_available() -> bool:
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg

        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        shim_dir = Path(gettempdir()) / "voice_assistant_ffmpeg_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / "ffmpeg.exe"
        if not shim_path.exists():
            shutil.copy2(ffmpeg_exe, shim_path)
        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def synthesize_mp3(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gTTS(text=text, lang="en").save(str(output_path))


def load_whisper_audio(audio_path: Path) -> np.ndarray:
    return whisper.load_audio(str(audio_path)).astype(np.float32)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wav.write(str(path), sample_rate, (clipped * 32767).astype(np.int16))


def evaluate(
    utterances: list[dict],
    output_dir: Path,
    report_path: Path,
    whisper_model_name: str,
    snr_levels: list[float | None],
    noise_seed: int,
) -> None:
    if not ensure_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required for Whisper audio loading. Install ffmpeg or imageio-ffmpeg."
        )

    np.random.seed(noise_seed)
    model = whisper.load_model(whisper_model_name)
    rows: list[dict] = []

    for item in utterances:
        utterance_id = item["id"]
        ground_truth = item["text"]
        clean_mp3 = output_dir / "clean" / f"{utterance_id}.mp3"
        synthesize_mp3(ground_truth, clean_mp3)
        clean_audio = load_whisper_audio(clean_mp3)

        for snr in snr_levels:
            if snr is None:
                label = "clean"
                test_audio = clean_audio
            else:
                label = f"snr_{int(snr)}db"
                test_audio = add_white_noise_snr(clean_audio, snr)

            test_wav = output_dir / label / f"{utterance_id}.wav"
            save_wav(test_wav, test_audio)

            result = model.transcribe(str(test_wav), language="en", fp16=False)
            prediction = result.get("text", "").strip()
            wer_score = wer(clean_text(ground_truth), clean_text(prediction))

            row = {
                "id": utterance_id,
                "ground_truth": ground_truth,
                "service": item.get("service", ""),
                "noise_condition": label,
                "snr_db": "" if snr is None else snr,
                "prediction": prediction,
                "wer": round(wer_score, 4),
            }
            rows.append(row)
            print(f"[{utterance_id} | {label}] wer={row['wer']:.4f}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    all_wer = [r["wer"] for r in rows]
    clean_wer = [r["wer"] for r in rows if r["noise_condition"] == "clean"]
    pass_gate_label = f"snr_{int(PASS_GATE_SNR)}db"
    pass_gate_wer = [r["wer"] for r in rows if r["noise_condition"] == pass_gate_label]

    mean_wer = statistics.mean(all_wer)
    mean_pass_gate_wer = statistics.mean(pass_gate_wer) if pass_gate_wer else None
    mean_clean_wer = statistics.mean(clean_wer) if clean_wer else None

    if mean_pass_gate_wer is None:
        raise ValueError(
            f"Pass gate SNR ({int(PASS_GATE_SNR)} dB) was not evaluated. "
            "Include 15 dB in --snr-levels or use default run settings."
        )

    passed = mean_pass_gate_wer < 0.20

    print("\n=== ASR BASELINE SUMMARY ===")
    print(f"Utterances: {len(utterances)}")
    print(f"Total runs: {len(rows)}")
    print(f"Mean WER (all conditions): {mean_wer:.4f}")
    print(
        f"Mean WER (pass gate: {int(PASS_GATE_SNR)} dB SNR): "
        f"{mean_pass_gate_wer:.4f}"
    )
    if mean_clean_wer is not None:
        print(f"Mean WER (clean reference only): {mean_clean_wer:.4f}")
    print(
        f"Pass gate (noisy WER < 0.20 at {int(PASS_GATE_SNR)} dB SNR): "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"Report: {report_path}")

    for label in sorted({r["noise_condition"] for r in rows}):
        values = [r["wer"] for r in rows if r["noise_condition"] == label]
        print(f"  {label}: mean WER {statistics.mean(values):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English ASR baseline with noisy WER.")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to utterance CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit utterances (0 = all in-scope safe rows).",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model name (default: small).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(VA_ROOT / "outputs" / "english" / "asr_baseline_audio"),
        help="Folder for generated audio.",
    )
    parser.add_argument(
        "--report-csv",
        default=str(VA_ROOT / "outputs" / "english" / "asr_baseline_reports" / "asr_wer_results.csv"),
        help="CSV report output path.",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Also run clean audio as reference (not used for pass gate).",
    )
    parser.add_argument(
        "--explore-snr",
        action="store_true",
        help="Evaluate 20/15/10 dB SNR for comparison. Pass gate still uses 15 dB only.",
    )
    parser.add_argument(
        "--snr-levels",
        type=float,
        nargs="+",
        help="Custom SNR levels in dB (must include 15 for pass gate).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=42,
        help="Random seed for noise generation (default: 42).",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    utterances = load_test_utterances(csv_path=args.csv, limit=limit)
    if not utterances:
        raise ValueError("No utterances loaded.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    report_path = Path(args.report_csv).with_name(f"asr_wer_results_{timestamp}.csv")

    if args.snr_levels:
        snr_levels = list(args.snr_levels)
    elif args.explore_snr:
        snr_levels = list(EXPLORATORY_SNR_LEVELS)
    else:
        snr_levels = [PASS_GATE_SNR]
    if args.include_clean:
        snr_levels = [None] + snr_levels

    evaluate(
        utterances=utterances,
        output_dir=output_dir,
        report_path=report_path,
        whisper_model_name=args.whisper_model,
        snr_levels=snr_levels,
        noise_seed=args.noise_seed,
    )


if __name__ == "__main__":
    main()
