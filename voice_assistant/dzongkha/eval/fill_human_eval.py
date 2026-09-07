"""One-off helper: fill human_eval_asr.csv from an offline WER report (snr_15db rows)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

VA_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    report = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        VA_ROOT / "outputs/dzongkha/asr/dzongkha_mms_wer_results_20260612_230205.csv"
    )
    eval_ts = sys.argv[2] if len(sys.argv) > 2 else "20260612_230205"
    noise = sys.argv[3] if len(sys.argv) > 3 else "snr_15db"

    out = Path(__file__).parent / "human_eval_asr.csv"
    audio_base = VA_ROOT / "outputs/dzongkha/asr/eval_audio" / eval_ts / noise

    rows_out = []
    with report.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["noise_condition"] != noise:
                continue
            uid = row["id"]
            wav = audio_base / f"{uid}.wav"
            try:
                audio_file = wav.relative_to(VA_ROOT).as_posix()
            except ValueError:
                audio_file = wav.as_posix()
            rows_out.append(
                {
                    "sample_id": str(int(uid.replace("D", ""))),
                    "audio_file": audio_file,
                    "reference_text_dzo": row["ground_truth"],
                    "mms_prediction_dzo": row["prediction"],
                    "reviewer_name": "",
                    "transcription_correct": "",
                    "language_is_dzongkha": "",
                    "understandable_1to5": "",
                    "notes": f"eval {eval_ts} {noise}; wer={row['wer']} cer={row['cer']}",
                }
            )

    fieldnames = [
        "sample_id",
        "audio_file",
        "reference_text_dzo",
        "mms_prediction_dzo",
        "reviewer_name",
        "transcription_correct",
        "language_is_dzongkha",
        "understandable_1to5",
        "notes",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {out}")


if __name__ == "__main__":
    main()
