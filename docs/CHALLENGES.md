# What broke (and what I changed)

---

## Whisper `base` missed words

Public-service phrases came back incomplete. Accuracy scripts (`English_asr.py`, `English_wer.py`, the noisy baseline) moved to Whisper `small`. The Edge TTS demo kept `base` so the full loop stayed quicker.

---

## Silence still went to Whisper

Without VAD, quiet buffers still cost latency and produced junk text. Silero `get_speech_timestamps` in `English_asr.py` and `English_tts.py` skips Whisper when nobody spoke.

On the mic demo in `voice_assistant/`, Silero also needed **at least 512 samples** at 16 kHz. A 30 ms chunk (480 samples) failed immediately. `voice_capture.py` uses 512-sample chunks.

---

## Two ways to stop recording

`English_asr.py` / `English_tts.py`: press Enter. `English_wer.py`: volume drop plus a few seconds of silence. Open questions need a manual stop; a known test sentence has a predictable end.

---

## LibriSpeech baseline needs a GPU

`asr_eng_baseline_metrics_dummyLibriSpeech.py` hardcodes `device="cuda"`. It will not rerun on CPU as written. The result file in `voice_module/asr/` is the GPU run.

---

## Windows OpenMP (`libiomp5md.dll`)

PyTorch (Silero) and NumPy/SciPy both load OpenMP. The voice demo sets `KMP_DUPLICATE_LIB_OK=TRUE` at startup. If it still crashes:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

---

## Output files landed in the wrong folder

Early paths were relative to whatever directory I ran from. Demo and eval scripts now root outputs at `voice_assistant/outputs/` via `Path(__file__)`.

---

## gTTS vs Coqui setup

Coqui needed extra packages and a first-run model download. ffmpeg was required for proxy WER (Whisper reading TTS audio). gTTS was enough for the English loop; Coqui stayed the offline backup. Proxy WER only checks whether Whisper can read the audio back.

---

## English noisy WER used synthesized speech

I needed known text and controlled 15 dB noise, so the baseline speaks the CSV with gTTS, then mixes noise. That passed (14.18% on 60 sentences). It is not the same as a live mic of those sentences.

---

## Groq answers without documents

The English Groq demos use a short system prompt only. They can sound sure and still be wrong on a specific Bhutan office rule. Grounded answers were Team A’s RAG work.

---

## Dzongkha

Whisper is not a Dzongkha ASR model (`dzo`, not Tibetan `bod`). Naive WER also breaks unless you split on tsheg (`་`).

`DZ_asr.py` needs `HF_TOKEN`. Local MMS downloads ~1 GB and uses the same adapter. `DZ_tts.py` needs Gemini and Google Translate keys. Dataset download for the eval: `Jyoti-77/dzongkha-asr`.
