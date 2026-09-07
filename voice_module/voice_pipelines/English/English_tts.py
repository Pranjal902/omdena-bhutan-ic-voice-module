import asyncio
import csv
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import edge_tts
import numpy as np
import pygame
import scipy.io.wavfile as wav
import sounddevice as sd
import torch
import whisper
from dotenv import load_dotenv
from groq import Groq
from silero_vad import get_speech_timestamps, load_silero_vad

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WHISPER_MODEL_SIZE = "base"
SAMPLE_RATE = 16000
CSV_FILE = "voice_pipeline_results.csv"

SYSTEM_PROMPT = """
You are a STRICT Bhutan Business Registration assistant.

Rules:
- Answer only questions related to Bhutan business registration,
  licensing, permits, company registration, taxation,
  and government business services.
- Keep responses concise and professional.
- If a question is unrelated, politely state that you only assist with Bhutan Business Registration and Licensing.
- Do not invent government policies.
- If information is unavailable, clearly say that you do not know.
"""

print("Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

print("Loading Silero VAD...")
vad_model = load_silero_vad()

print("Connecting Groq...")
client = Groq(api_key=GROQ_API_KEY)

pygame.mixer.init()

print("System Ready\n")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Timestamp", "User_Query", "ASR_Time", "LLM_Time", "TTS_Time", "Total_E2E_Time"]
        )

audio_queue = queue.Queue()
recording = False


def audio_callback(indata, frames, time_info, status):
    global recording
    if recording:
        audio_queue.put(indata.copy())


def stop_recording():
    global recording
    input("\nPress ENTER to stop recording...\n")
    recording = False


def record_audio():
    global recording

    while not audio_queue.empty():
        audio_queue.get()

    input("\nPress ENTER to start recording...\n")

    recording = True
    full_audio = []

    print("Recording...")

    threading.Thread(target=stop_recording, daemon=True).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback
    ):
        while True:
            if not recording and audio_queue.empty():
                break

            try:
                chunk = audio_queue.get(timeout=0.1)
                chunk = chunk.flatten()
                full_audio.extend(chunk)

            except queue.Empty:
                continue

    return np.array(full_audio, dtype=np.float32)


def speech_detected(audio_np):
    audio_tensor = torch.from_numpy(audio_np).float()

    speech = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=SAMPLE_RATE)

    return len(speech) > 0


def speech_to_text(audio_np):
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    temp_wav.close()

    wav.write(temp_wav.name, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))

    start = time.time()

    result = whisper_model.transcribe(temp_wav.name, language="en", fp16=False)

    end = time.time()

    os.remove(temp_wav.name)

    return result["text"].strip(), end - start


def ask_llm(user_query):
    start = time.time()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content

    except Exception as e:
        answer = f"Error: {e!s}"

    end = time.time()

    return answer, end - start


def stop_speaking(stop_event):
    input("\nPress ENTER to stop speaking...\n")
    stop_event.set()
    pygame.mixer.music.stop()


def speak(text):
    start = time.time()

    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)

    filename = temp_mp3.name
    temp_mp3.close()

    async def generate_tts():
        communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
        await communicate.save(filename)

    asyncio.run(generate_tts())

    stop_event = threading.Event()

    threading.Thread(target=stop_speaking, args=(stop_event,), daemon=True).start()

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        if stop_event.is_set():
            break

        time.sleep(0.1)

    pygame.mixer.music.stop()

    try:
        pygame.mixer.music.unload()
    except:
        pass

    time.sleep(0.2)

    try:
        os.remove(filename)
    except:
        pass

    return time.time() - start

    return end - start


def log_results(query, asr_time, llm_time, tts_time, total_time):
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                query,
                round(asr_time, 2),
                round(llm_time, 2),
                round(tts_time, 2),
                round(total_time, 2),
            ]
        )


print("=" * 60)
print("BHUTAN BUSINESS REGISTRATION VOICE ASSISTANT")
print("=" * 60)

while True:
    total_start = time.time()

    audio_np = record_audio()

    if len(audio_np) == 0:
        print("No audio recorded")
        continue

    if not speech_detected(audio_np):
        print("No speech detected")
        continue

    user_text, asr_time = speech_to_text(audio_np)

    print("\nUSER:")
    print(user_text)

    if user_text.strip().lower() in ["exit", "quit", "stop"]:
        print("Goodbye")
        break

    answer, llm_time = ask_llm(user_text)

    print("\nASSISTANT:")
    print(answer)

    tts_time = speak(answer)

    total_time = time.time() - total_start

    print("\n========== LATENCY ==========")
    print(f"ASR Time: {asr_time:.2f} sec")
    print(f"LLM Time: {llm_time:.2f} sec")
    print(f"TTS Time: {tts_time:.2f} sec")
    print(f"Total E2E Time: {total_time:.2f} sec")

    log_results(user_text, asr_time, llm_time, tts_time, total_time)

    choice = input("\nType 'continue' to ask another question or 'quit' to exit: ").strip().lower()

    if choice == "quit":
        print("Goodbye")
        break
