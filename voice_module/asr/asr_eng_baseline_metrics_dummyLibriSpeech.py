# ASR baseline for English

import time

import soundfile as sf
from datasets import load_dataset
from faster_whisper import WhisperModel
from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer

# Load small librispeech_asr_dummy dataset from HuggingFace
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load WHISPER models: tiny & medium
model1 = WhisperModel("tiny", device="cuda", compute_type="float16")

model2 = WhisperModel("medium", device="cuda", compute_type="float16")

# Perform text normalization
basic_transform = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

number_map = {"20s": "twenties", "30s": "thirties", "40s": "forties", "50s": "fifties"}


def normalize_numbers(text):
    for k, v in number_map.items():
        text = text.replace(k, v)
    return text


# ASR

references = []
predictions1 = []
predictions2 = []
total_processing_time1 = 0.0
total_processing_time2 = 0.0
total_audio_duration = 0.0

for sample in dataset:
    audio_array = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sample_rate
    total_audio_duration += audio_duration

    # save temporary wav
    sf.write("temp.wav", audio_array, sample_rate, subtype="PCM_16")

    # ===========================================
    # Experiment with tiny model
    # note execution time:
    start = time.time()

    # transcribe
    segments1, info1 = model1.transcribe("temp.wav", language="en")
    segments1 = list(segments1)

    end = time.time()
    processing_time = end - start
    total_processing_time1 += processing_time

    predicted_text1 = " ".join(seg.text.strip() for seg in segments1)

    reference_text = sample["text"]

    predictions1.append(predicted_text1)
    references.append(reference_text)

    # ===========================================
    # Experiment with medium model
    start2 = time.time()

    # transcribe
    segments2, info2 = model2.transcribe("temp.wav", language="en")
    segments2 = list(segments2)

    end2 = time.time()
    processing_time2 = end2 - start2
    total_processing_time2 += processing_time2

    predicted_text2 = " ".join(seg.text.strip() for seg in segments2)

    predictions2.append(predicted_text2)

# apply normalizations
normalized_refs = []
normalized_preds1 = []
normalized_preds2 = []

for ref, pred1, pred2 in zip(references, predictions1, predictions2):
    ref = basic_transform(ref)

    transformed_pred1 = basic_transform(pred1)
    transformed_pred2 = basic_transform(pred2)

    ref = normalize_numbers(ref)

    norm_pred1 = normalize_numbers(transformed_pred1)
    norm_pred2 = normalize_numbers(transformed_pred2)

    normalized_refs.append(ref)

    normalized_preds1.append(norm_pred1)
    normalized_preds2.append(norm_pred2)


# Compute Metrics: WER, Real-Time Factor (RTF), Speedup

final_wer1 = wer(normalized_refs, normalized_preds1)
rtf1 = total_processing_time1 / total_audio_duration
speedup1 = 1 / rtf1

final_wer2 = wer(normalized_refs, normalized_preds2)
rtf2 = total_processing_time2 / total_audio_duration
speedup2 = 1 / rtf2
