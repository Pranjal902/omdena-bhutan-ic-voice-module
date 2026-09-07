"""gTTS synthesis helpers for English TTS evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import Optional

from gtts import gTTS
from gtts.tts import gTTSError


@dataclass
class SynthesisResult:
    text: str
    language: str
    output_path: str
    elapsed_seconds: float
    file_size_bytes: int
    status: str
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def synthesize_text(
    text: str,
    output_path: str | Path,
    language: str = "en",
    tld: str = "com",
    slow: bool = False,
    timeout_seconds: Optional[float] = 15,
) -> SynthesisResult:
    """Synthesize text to MP3 and capture runtime metrics."""
    if not text.strip():
        raise ValueError("Input text must not be empty.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    try:
        tts = gTTS(text=text, lang=language, tld=tld, slow=slow, timeout=timeout_seconds)
        tts.save(str(output))
        elapsed = time.perf_counter() - started
        file_size = output.stat().st_size if output.exists() else 0
        return SynthesisResult(
            text=text,
            language=language,
            output_path=str(output),
            elapsed_seconds=round(elapsed, 4),
            file_size_bytes=file_size,
            status="success",
        )
    except (gTTSError, Exception) as exc:
        elapsed = time.perf_counter() - started
        return SynthesisResult(
            text=text,
            language=language,
            output_path=str(output),
            elapsed_seconds=round(elapsed, 4),
            file_size_bytes=0,
            status="failed",
            error_message=str(exc),
        )
