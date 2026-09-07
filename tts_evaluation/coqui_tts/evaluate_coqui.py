"""Evaluate Coqui TTS on latency, reliability, and intelligibility proxy."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import csv
import statistics
import math
import os
import shutil
from tempfile import gettempdir

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from jiwer import wer
import whisper

from coqui_engine import DEFAULT_MODEL, load_model, synthesize_text
from sample_prompts import DEFAULT_EVAL_SENTENCES


def ensure_ffmpeg_available() -> bool:
    """Ensure ffmpeg is available in PATH for Whisper."""
    if shutil.which("ffmpeg"):
        return True

    try:
        import imageio_ffmpeg

        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        shim_dir = Path(gettempdir()) / "coqui_ffmpeg_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / "ffmpeg.exe"

        if not shim_path.exists():
            shutil.copy2(ffmpeg_exe, shim_path)

        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False


def clean_text(text: str) -> str:
    allowed = []
    for char in text.lower():
        if char.isalnum() or char.isspace():
            allowed.append(char)
    return "".join(allowed).strip()


def run_evaluation(
    whisper_model_name: str,
    coqui_model_name: str,
    output_dir: Path,
    csv_path: Path,
    repeats: int,
    skip_asr: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Coqui model once before evaluation...")
    model_load_seconds = load_model(model_name=coqui_model_name)
    print(f"Coqui model load time: {model_load_seconds:.4f} sec")

    whisper_model = None
    asr_enabled = not skip_asr
    if asr_enabled:
        if ensure_ffmpeg_available():
            whisper_model = whisper.load_model(whisper_model_name)
        else:
            asr_enabled = False
            print(
                "ASR scoring disabled: ffmpeg not found. Install ffmpeg or use "
                "`pip install imageio-ffmpeg`."
            )

    rows = []
    for repeat_index in range(repeats):
        for sentence_index, sentence in enumerate(DEFAULT_EVAL_SENTENCES, start=1):
            safe_index = f"s{sentence_index:02d}_r{repeat_index+1:02d}"
            audio_path = output_dir / f"{safe_index}.wav"
            synth_result = synthesize_text(
                text=sentence,
                output_path=audio_path,
                model_name=coqui_model_name,
            )

            if synth_result.status == "success" and asr_enabled and whisper_model is not None:
                try:
                    transcript = whisper_model.transcribe(str(audio_path)).get("text", "")
                    wer_score = wer(clean_text(sentence), clean_text(transcript))
                except FileNotFoundError as exc:
                    transcript = ""
                    wer_score = None
                    asr_enabled = False
                    print(
                        "ASR scoring disabled: ffmpeg is not installed or not in PATH. "
                        f"Details: {exc}"
                    )
            else:
                transcript = ""
                wer_score = None if synth_result.status == "success" else 1.0

            row = {
                "repeat": repeat_index + 1,
                "sentence_id": sentence_index,
                "input_text": sentence,
                "coqui_model": coqui_model_name,
                "status": synth_result.status,
                "latency_seconds": synth_result.elapsed_seconds,
                "file_size_bytes": synth_result.file_size_bytes,
                "asr_prediction": transcript,
                "proxy_wer": round(wer_score, 4) if wer_score is not None else "",
                "error_message": synth_result.error_message,
            }
            rows.append(row)
            print(
                f"[{safe_index}] status={row['status']} "
                f"latency={row['latency_seconds']:.4f}s wer={row['proxy_wer']}"
            )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    successful = [r for r in rows if r["status"] == "success"]
    latency_values = [r["latency_seconds"] for r in successful]
    wer_values = [r["proxy_wer"] for r in successful if isinstance(r["proxy_wer"], float)]
    success_rate = len(successful) / len(rows)

    print("\n=== COQUI TTS EVALUATION SUMMARY ===")
    print(f"Coqui model load time (once): {model_load_seconds:.4f} sec")
    print(f"Total runs: {len(rows)}")
    print(f"Success rate: {success_rate*100:.2f}%")
    if latency_values:
        print(f"Mean latency: {statistics.mean(latency_values):.4f} sec")
        p95_index = max(0, math.ceil(0.95 * len(latency_values)) - 1)
        print(f"P95 latency: {sorted(latency_values)[p95_index]:.4f} sec")
    if wer_values:
        print(f"Mean proxy WER: {statistics.mean(wer_values):.4f}")
    else:
        print("Mean proxy WER: not available (ASR scoring disabled)")
    print(f"CSV report: {csv_path}")
    print("Note: proxy WER is an intelligibility indicator, not human MOS quality.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Coqui English TTS evaluation.")
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model for proxy intelligibility scoring (default: base).",
    )
    parser.add_argument(
        "--coqui-model",
        default=DEFAULT_MODEL,
        help=f"Coqui TTS model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many times to run the full sentence set (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        default="tts_evaluation/coqui_tts/outputs/coqui_eval_audio",
        help="Folder for generated audio files.",
    )
    parser.add_argument(
        "--report-csv",
        default="tts_evaluation/coqui_tts/outputs/coqui_eval_reports/coqui_eval_results.csv",
        help="CSV output path for full run report.",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="Skip Whisper transcription and WER scoring.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    report_path = Path(args.report_csv).with_name(f"coqui_eval_results_{timestamp}.csv")

    run_evaluation(
        whisper_model_name=args.whisper_model,
        coqui_model_name=args.coqui_model,
        output_dir=output_dir,
        csv_path=report_path,
        repeats=args.repeats,
        skip_asr=args.skip_asr,
    )


if __name__ == "__main__":
    main()
