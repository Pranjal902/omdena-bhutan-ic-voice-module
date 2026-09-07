import os
import tempfile
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from silero_vad import collect_chunks, get_speech_timestamps, load_silero_vad

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")

SAMPLE_RATE = 16000
DURATION = 8

vad_model = load_silero_vad()

audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()

audio = np.squeeze(audio)

audio_int16 = (audio * 32768).astype(np.int16)

speech_timestamps = get_speech_timestamps(audio_int16, vad_model, sampling_rate=SAMPLE_RATE)

if len(speech_timestamps) == 0:
    print("No speech detected")
    exit()

clean_audio_int16 = collect_chunks(speech_timestamps, audio_int16)

clean_audio = clean_audio_int16.astype(np.float32) / 32768.0

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    sf.write(f.name, clean_audio, SAMPLE_RATE)
    audio_path = f.name

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

with open(audio_path, "rb") as file:
    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/mms-1b-all",
        headers=headers,
        data=file,
    )

print(response.json())

os.remove(audio_path)
