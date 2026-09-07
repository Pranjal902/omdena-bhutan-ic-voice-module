"""Load public-service English utterances for ASR baseline."""

from __future__ import annotations

from pathlib import Path
import csv

DEFAULT_CSV = Path(__file__).resolve().parent / "bhutan_classifier_test_set.csv"


def load_test_utterances(
    csv_path: str | Path = DEFAULT_CSV,
    in_scope_only: bool = True,
    safe_only: bool = True,
    limit: int | None = None,
) -> list[dict]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Utterance CSV not found: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if in_scope_only and row.get("in_scope") != "in_scope":
                continue
            if safe_only and row.get("safety") != "safe":
                continue
            rows.append(
                {
                    "id": row["id"],
                    "text": row["user_question"].strip(),
                    "service": row.get("service", ""),
                    "request_type": row.get("request_type", ""),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows
