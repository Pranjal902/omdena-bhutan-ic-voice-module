"""Whisper transcription helpers."""

from __future__ import annotations

import os

import whisper

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_model = None
_model_name = None


def load_whisper(model_name: str = "small"):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        _model = whisper.load_model(model_name)
        _model_name = model_name
    return _model


def transcribe_file(audio_path: str, model_name: str = "small") -> str:
    model = load_whisper(model_name)
    result = model.transcribe(audio_path, language="en", fp16=False)
    return result.get("text", "").strip()
