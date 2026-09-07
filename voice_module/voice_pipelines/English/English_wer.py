import csv
import os
import queue
import re
import tempfile

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import whisper
from jiwer import wer

model = whisper.load_model("small")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 4

csv_file = "wer_results.csv"

if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Ground Truth", "Prediction", "WER"])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


test_sentences = [
    "I want to apply for land registration",
    "I need a business license",
    "Where is the tax office",
    "How can I renew my permit",
    "I want to open a shop in Thimphu",
    "Where is the nearest hospital in Paro",
    "I need help with passport renewal",
    "Can I register my business online",
    "How do I apply for a driving license",
    "What documents are needed for visa approval",
    "I live in Punakha",
    "I am traveling to Wangdue Phodrang",
    "Show me government offices in Thimphu",
    "How can I pay electricity bills",
    "Where can I renew my citizenship card",
    "I want information about land tax",
    "How do I update my phone number",
    "I need water supply registration",
    "Where is the police station in Gelephu",
    "I want to apply for a construction permit",
    "Can I get business approval online",
    "Where is immigration office",
    "How can I check my application status",
    "I need municipal services",
    "Where is the nearest bank in Bhutan",
    "I want internet connection registration",
    "How do I file a complaint",
    "Where is customs office",
    "Can I update my address online",
    "How do I renew trade license",
    "I want to visit Trongsa",
    "Show me services available in Haa",
    "I need support for tax payment",
    "Where can I get health insurance",
    "How do I apply for a birth certificate",
    "Can I renew my permit online",
    "I need road transport services",
    "Where is labor office",
    "How can I contact municipality office",
    "I need education scholarship details",
    "Where is airport in Paro",
    "How do I check land ownership",
    "I need electricity connection",
    "Can I apply for loan services",
    "I want to register my company",
    "Where is district office in Mongar",
    "How do I get a new passport",
    "I need vehicle registration services",
    "Where is tourism office",
    "I want digital government services",
]


def record_until_silence():

    print("\nStart speaking...")
    print("Recording will stop after 2 seconds of silence")

    audio_queue = queue.Queue()

    recorded_audio = []

    silence_counter = 0

    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback):
        while True:
            data = audio_queue.get()

            recorded_audio.append(data)

            volume = np.abs(data).mean()

            if volume < SILENCE_THRESHOLD:
                silence_counter += 1
            else:
                silence_counter = 0

            if silence_counter > (SILENCE_DURATION * 10):
                break

    audio_np = np.concatenate(recorded_audio, axis=0)

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    wav.write(temp_wav.name, SAMPLE_RATE, audio_np)

    print("Recording stopped")

    return temp_wav.name


def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]


all_wer = []

for idx, ground_truth in enumerate(test_sentences, start=1):
    print("\n========================")
    print(f"Sentence {idx}")
    print("========================")

    print("Read this sentence:")
    print(ground_truth)

    input("\nPress Enter to start...")

    audio_file = record_until_silence()

    prediction = speech_to_text(audio_file)

    gt_clean = clean_text(ground_truth)
    pred_clean = clean_text(prediction)

    error = wer(gt_clean, pred_clean)

    all_wer.append(error)

    print("\nPrediction:")
    print(prediction)

    print("\nWER:")
    print(error)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([ground_truth, prediction, round(error, 4)])

    os.remove(audio_file)

    choice = input("\nContinue? (y/n): ").lower()

    if choice != "y":
        break

average_wer = sum(all_wer) / len(all_wer)

print("\n========================")
print("FINAL RESULTS")
print("========================")

print(f"Total Sentences Tested: {len(all_wer)}")
print(f"Average WER: {average_wer}")
