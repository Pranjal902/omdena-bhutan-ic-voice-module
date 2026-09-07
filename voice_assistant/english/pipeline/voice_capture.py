"""Live microphone capture with Silero VAD end-of-speech detection."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import queue

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import torch

SAMPLE_RATE = 16000
# Silero VAD requires >= 512 samples at 16 kHz (32 ms). 30 ms = 480 samples → "chunk too short".
CHUNK_SAMPLES = 512
SILENCE_MS = 900
MAX_RECORD_SECONDS = 20


def _load_vad_model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def record_utterance(output_dir: str | Path) -> Path:
    """Record one user utterance and stop after short silence."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    vad_model = _load_vad_model()
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    recorded: list[np.ndarray] = []

    chunk_ms = 1000 * CHUNK_SAMPLES / SAMPLE_RATE
    silence_chunks = max(1, int(SILENCE_MS / chunk_ms))
    max_chunks = int(MAX_RECORD_SECONDS * 1000 / chunk_ms)

    silent_run = 0
    speech_started = False

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    print("Speak now... (say your question, then pause briefly when done)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=callback,
    ):
        for _ in range(max_chunks):
            chunk = audio_queue.get().flatten()
            recorded.append(chunk)

            tensor = torch.from_numpy(chunk)
            prob = vad_model(tensor, SAMPLE_RATE).item()

            if prob > 0.5:
                speech_started = True
                silent_run = 0
            elif speech_started:
                silent_run += 1
                if silent_run >= silence_chunks:
                    break

    if not speech_started:
        raise RuntimeError(
            "No speech detected. Check your microphone, then try again and speak right after 'Speak now...'."
        )

    audio = np.concatenate(recorded).astype(np.float32)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = output / f"user_input_{timestamp}.wav"
    wav.write(str(wav_path), SAMPLE_RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    print("Recording saved:", wav_path)
    return wav_path
