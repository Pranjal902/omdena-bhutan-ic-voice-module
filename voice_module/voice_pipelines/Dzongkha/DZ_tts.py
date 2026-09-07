import os
import threading
from pathlib import Path

import requests
import sounddevice as sd
import torch
from dotenv import load_dotenv
from google import genai
from transformers import AutoTokenizer, VitsModel

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GOOGLE_TRANSLATE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")

SYSTEM_PROMPT = """
You are a Business Registration and Licensing Assistant for Bhutan.

Responsibilities:
- Help users with business registration.
- Help users with licensing procedures.
- Explain required documents and processes.
- Give clear and helpful answers.

Rules:
- Answer in English only.
- Keep responses concise and accurate.
"""

print("Loading Dzongkha TTS model...")

tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-dzo")

tts_model = VitsModel.from_pretrained("facebook/mms-tts-dzo")

print("TTS model loaded successfully.")


def translate_text(text, source_lang, target_lang):
    url = f"https://translation.googleapis.com/language/translate/v2?key={GOOGLE_TRANSLATE_API_KEY}"

    payload = {"q": text, "source": source_lang, "target": target_lang, "format": "text"}

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["data"]["translations"][0]["translatedText"]


def speak(text):
    tts_inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        waveform = tts_model(**tts_inputs).waveform

    audio = waveform.squeeze().cpu().numpy()

    sd.play(audio, samplerate=tts_model.config.sampling_rate)


def stop_on_enter():
    input("\nPress ENTER to stop speaking...")
    sd.stop()


while True:
    dz_question = input("\nAsk in Dzongkha: ").strip()

    if dz_question in ["མཇུག", "བཞག", "ཕྱིར་འགྱོ"]:
        goodbye = "བཀའ་དྲིན་ཆེ། ལེགས་སོ།"

        print("\nDzongkha Answer:")
        print(goodbye)

        speak(goodbye)
        sd.wait()

        break

    if not dz_question:
        continue

    english_question = translate_text(dz_question, "dz", "en")

    prompt = f"""
{SYSTEM_PROMPT}

User Question:
{english_question}
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    english_answer = response.text.strip()

    dz_answer = translate_text(english_answer, "en", "dz")

    print("\nDzongkha Answer:")
    print(dz_answer)

    speak(dz_answer)

    stopper = threading.Thread(target=stop_on_enter, daemon=True)

    stopper.start()

    sd.wait()
