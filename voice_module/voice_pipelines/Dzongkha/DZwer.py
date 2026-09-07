import csv
import os
import queue
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
    return text.strip()


test_sentences = [
    "ས་ཆའི་ཐོ་བཀོད་ཀྱི་དོན་ལུ་ཞུ་བ་འབད་དགོ་མནོཝ་མས།",
    "ང་ལུ་ཚོང་ལས་ཆོག་ཐམ་དགོཔ་ཨིན།",
    "ཁྲལ་ཡིག་ཚང་ག་ཏེ་ཡོད།",
    "ངའི་ཆོག་མཆན་བསྐྱར་གསོ་གང་འདྲ་བྱེད་ཐུབ་བམ།",
    "ཐིམ་ཕུག་ལུ་ ཚོང་ཁང་སྒོ་ཕྱེ་དགོ་མནོཝ་མས།སྤ་རོ་ལུ་སྨན་ཁང་ཉེ་ཤོས་ག་ཏེ་སྨོ",
    "ང་ལུ་ཕྱི་སྐྱོད་ལག་ཁྱེར་བསྐྱར་གསོ་འབད་ནི་ལུ་གྲོགས་རམ་དགོཔ་ཨིན།",
    "ངེ་གི་ཚོང་འབྲེལ་འདི་ ཡོངས་འབྲེལ་ཐོག་ལས་ ཐོ་བཀོད་འབད་ཚུགས་ག",
    "སྣུམ་འཁོར་བཏང་ཆོག་པའི་ལག་ཁྱེར་ག་དེ་སྦེ་ཞུ་ནི་ཨིན་ན།",
    "སྐྱབས་བཅོལ་ལག་ཁྱེར་ཆ་འཇོག་འབད་ནིའི་དོན་ལུ་ ཡིག་ཆ་ག་ཅི་དགོཔ་སྨོ",
    "ང་སྤུ་ན་ཁ་ལུ་སྡོདཔ་ཨིན།",
    "ང་དབང་འདུས་ཕོ་བྲང་ལུ་འགྲོ་འགྲུལ་འབད་དོ།",
    "ཐིམ་ཕུག་ལུ་ གཞུང་གི་ཡིག་ཚང་ཚུ་སྟོན་གནང་།",
    "གློག་རིན་ག་དེ་སྦེ་སྤྲོད་ཚུགས་ག",
    "མི་ཁུངས་ལག་ཁྱེར་ག་ཏེ་ལས་བསྐྱར་གསོ་འབད་ཚུགས་ག",
    "ང་ལུ་ས་ཆའི་ཁྲལ་གྱི་སྐོར་ལས་བརྡ་དོན་དགོཔ་འདུག",
    "ངེ་གི་ཁ་པར་ཨང་གྲངས་དུས་མཐུན་བཟོ་ཐངས།",
    "ང་ལུ་ཆུ་བཀྲམ་སྤེལ་གྱི་ཐོ་བཀོད་དགོཔ་ཨིན།",
    "དགེ་ལེ་ཕུག་ལུ་ ཁྲིམས་སྲུང་འགག་སྡེ་ག་ཏེ་ཡོདཔ་སྨོ",
    "ང་བཟོ་བསྐྲུན་ཆོག་ཐམ་གྱི་ཞུ་ཡིག་བཙུགས་དགོ་མནོཝ་མས།",
    "ང་གིས་ཡོངས་འབྲེལ་ཐོག་ལས་ ཚོང་འབྲེལ་གྱི་ཆ་འཇོག་ཐོབ་ཚུགས་ག",
    "ཕྱི་མི་ནང་སྐྱོད་ཡིག་ཚང་ག་ཏེ་ཡོདཔ་སྨོ",
    "ངེ་གི་ཞུ་ཡིག་གནས་ཚད་ག་དེ་སྦེ་ཞིབ་དཔྱད་འབད་ཚུགས་ག",
    "ང་ལུ་ཁྲོམ་སྡེའི་ཞབས་ཏོག་དགོཔ་ཨིན།",
    "འབྲུག་གི་དངུལ་ཁང་ཉེ་ཤོས་ག་ཏེ་ཡོདཔ་སྨོ",
    "ང་ལུ་ཡོངས་འབྲེལ་མཐུད་ལམ་ཐོ་བཀོད་དགོཔ་འདུག",
    "ཉོགས་བཤད་ག་དེ་སྦེ་བཙུགས་ནི་ཨིན་ན",
    "ཅ་དམ་ཡིག་ཚང་ག་ཏེ་ཡོདཔ་སྨོ",
    "ངེ་གི་ཁ་བྱང་ཡོངས་འབྲེལ་ཐོག་ལས་དུས་མཐུན་བཟོ་ཚུགས་ག",
    "ཚོང་འབྲེལ་ཆོག་ཐམ་བསྐྱར་གསོ་འབད་ཐངས།",
    "ང་ཀྲོང་ས་ལུ་འགྱོ་དགོ་མནོཝ་མས།",
    "ཧ་ནང་ལུ་ཡོད་པའི་ཞབས་ཏོག་ཚུ་ང་ལུ་སྟོན",
    "ང་ལུ་ཁྲལ་སྤྲོད་ནིའི་དོན་ལུ་རྒྱབ་སྐྱོར་དགོཔ་ཨིན།",
    "གསོ་བའི་ཉེན་བཅོལ་གང་ནས་ཐོབ་ཐུབ་བམ།སྐྱེས་ཚེས་ལག་ཁྱེར་ཞུ་ཐབས།",
    "ང་གིས་ ངེ་གི་ཆོག་ཐམ་འདི་ ཡོངས་འབྲེལ་ཐོག་ལས་ བསྐྱར་གསོ་འབད་ཚུགས་ག",
    "ང་ལུ་ལམ་འགྲུལ་སྐྱེལ་འདྲེན་གྱི་ཞབས་ཏོག་དགོཔ་ཨིན།",
    "ལས་མི་ཡིག་ཚང་ག་ཏེ་ཡོདཔ་སྨོ",
    "ཁྲོམ་སྡེ་ཡིག་ཚང་དང་འབྲེལ་བ་འཐབ་ཐངས།",
    "ང་ལུ་ཤེས་རིག་སློབ་གཉེར་གྲོགས་རམ་གྱི་ཁ་གསལ་དགོཔ་ཨིན།",
    "སྤ་རོ་ལུ་གནམ་གྲུ་ཐང་ག་ཏེ་ཡོདཔ་སྨོ",
    "ས་ཆའི་བདག་དབང་ག་དེ་སྦེ་བརྟག་དཔྱད་འབད་ནི་ཨིན་ན",
    "ང་ལ་གློག་མཐུད་དགོས།",
    "སྐྱིན་འགྲུལ་གྱི་ཞབས་ཏོག་ཞུ་ཆོག་གམ།",
    "ངེ་གི་ལས་སྡེ་ཐོ་བཀོད་འབད་དགོ་མནོཝ་མས།",
    "མོང་སྒར་ལུ་རྫོང་ཁག་ཡིག་ཚང་ག་ཏེ་ཡོདཔ་སྨོ",
    "ཕྱི་སྐྱོད་ལག་ཁྱེར་གསརཔ་ག་དེ་སྦེ་ཐོབ་ཚུགས་ག",
    "ང་ལུ་སྣུམ་འཁོར་ཐོ་བཀོད་ཀྱི་ཞབས་ཏོག་དགོ",
    "ལྟ་བཤལ་ཡིག་ཚང་ག་ཏེ་ཡོདཔ་སྨོ",
    "ང་ལུ་དྲི་ཇི་ཊལ་གཞུང་གི་ཞབས་ཏོག་དགོཔ་ཨིན།",
]


def record_until_silence():

    print("\nStart speaking...")
    print("Recording will stop after 4 seconds of silence")

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
    result = model.transcribe(audio_path, language="dz")
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
