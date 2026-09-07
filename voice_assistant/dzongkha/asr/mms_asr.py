"""MMS ASR helpers for Dzongkha (ISO 639-3: dzo — not Tibetan bod)."""

from __future__ import annotations

import time
import unicodedata
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

MMS_MODEL_ID = "facebook/mms-1b-all"
DZO_LANG = "dzo"
SAMPLE_RATE = 16000
TSHEG = "\u0f0b"  # Tibetan mark intersyllable tsheg — useful WER word boundary


def normalize_dzo_text(text: str) -> str:
    """Light normalization for metric comparison (keep Tibetan script intact)."""
    text = unicodedata.normalize("NFC", text.strip())
    return " ".join(text.split())


def tokenize_for_wer(text: str) -> str:
    """Split Dzongkha into word-like units for jiwer (tsheg + whitespace)."""
    text = normalize_dzo_text(text)
    text = text.replace(TSHEG, " ")
    return " ".join(text.split())


def load_mms_dzo(device: str | None = None) -> tuple[Wav2Vec2ForCTC, AutoProcessor, str]:
    """Load MMS-1B with the Dzongkha adapter."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID)

    processor.tokenizer.set_target_lang(DZO_LANG)
    model.load_adapter(DZO_LANG)

    model.to(device)
    model.eval()
    return model, processor, device


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def resample_if_needed(audio: np.ndarray, sample_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    if sample_rate == target_rate:
        return _to_mono_float32(audio)
    try:
        import torchaudio.functional as F
        import torch as th

        tensor = th.from_numpy(_to_mono_float32(audio)).unsqueeze(0)
        resampled = F.resample(tensor, sample_rate, target_rate)
        return resampled.squeeze(0).numpy()
    except Exception:
        duration = len(audio) / sample_rate
        target_len = int(duration * target_rate)
        if target_len < 1:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        x_new = np.linspace(0, 1, num=target_len, endpoint=False)
        return np.interp(x_new, x_old, _to_mono_float32(audio)).astype(np.float32)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    sample_rate, audio = wav.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    return _to_mono_float32(audio), int(sample_rate)


def transcribe_array(
    audio: np.ndarray,
    sample_rate: int,
    model: Wav2Vec2ForCTC,
    processor: AutoProcessor,
    device: str,
) -> tuple[str, float]:
    """Return (transcription, processing_seconds)."""
    audio = resample_if_needed(audio, sample_rate)
    if audio.size == 0:
        return "", 0.0

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    elapsed = time.perf_counter() - start

    ids = torch.argmax(logits, dim=-1)[0]
    text = processor.decode(ids).strip()
    return text, elapsed


def transcribe_file(
    path: str | Path,
    model: Wav2Vec2ForCTC,
    processor: AutoProcessor,
    device: str,
) -> tuple[str, float]:
    audio, sample_rate = read_wav(path)
    return transcribe_array(audio, sample_rate, model, processor, device)
