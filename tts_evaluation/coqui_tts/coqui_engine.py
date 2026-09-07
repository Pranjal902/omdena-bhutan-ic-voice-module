"""Coqui TTS synthesis helpers for English TTS evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import Optional

DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

_tts_instance = None
_loaded_model_name: Optional[str] = None


@dataclass
class SynthesisResult:
    text: str
    model_name: str
    output_path: str
    elapsed_seconds: float
    file_size_bytes: int
    status: str
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def load_model(model_name: str = DEFAULT_MODEL, gpu: bool = False):
    """Load (or reload) the Coqui model. Returns load time in seconds."""
    global _tts_instance, _loaded_model_name

    started = time.perf_counter()
    from TTS.api import TTS

    if _tts_instance is None or _loaded_model_name != model_name:
        _tts_instance = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        _loaded_model_name = model_name

    return round(time.perf_counter() - started, 4)


def synthesize_text(
    text: str,
    output_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    gpu: bool = False,
) -> SynthesisResult:
    """Synthesize text to WAV and capture runtime metrics."""
    if not text.strip():
        raise ValueError("Input text must not be empty.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    try:
        load_model(model_name=model_name, gpu=gpu)
        _tts_instance.tts_to_file(text=text, file_path=str(output))
        elapsed = time.perf_counter() - started
        file_size = output.stat().st_size if output.exists() else 0
        return SynthesisResult(
            text=text,
            model_name=model_name,
            output_path=str(output),
            elapsed_seconds=round(elapsed, 4),
            file_size_bytes=file_size,
            status="success",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return SynthesisResult(
            text=text,
            model_name=model_name,
            output_path=str(output),
            elapsed_seconds=round(elapsed, 4),
            file_size_bytes=0,
            status="failed",
            error_message=str(exc),
        )
