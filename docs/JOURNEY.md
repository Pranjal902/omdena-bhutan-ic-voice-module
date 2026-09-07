# How this voice work unfolded

My team handled speech in and speech out for public-service questions, English and Dzongkha. How that sat next to NLP, agents, and integration is in the [root README](../README.md).

---

## English capture

First I needed a mic recording that ignored silence and still produced usable English text.

That is `voice_module/voice_pipelines/English/English_asr.py`. Record until Enter, run Silero VAD, send speech to Whisper `small`, append times to `latency_results.csv`. File runs sat around 7–21 seconds total, depending on how long I talked.

Whisper `base` dropped words on public-service phrases, so the accuracy scripts stayed on `small`. The spoken demo (`English_tts.py`) kept `base` so the loop felt faster. Without VAD, Whisper still ran on quiet buffers and wasted time.

---

## Checking the transcript

- Mic read-aloud — `English_wer.py` shows a sentence, I speak it, JiWER scores against the prompt. The first three rows in `wer_results.csv` are 0.0 WER.
- Offline dummy set — `asr/asr_eng_baseline_metrics_dummyLibriSpeech.py` compares faster-whisper tiny vs medium on Hugging Face’s LibriSpeech dummy. Tiny **11.21%** WER, medium **6.60%**. That script expects CUDA; the saved `.txt` is the GPU run.

---

## Spoken answers

The spoken Groq demo is `English_tts.py`:

mic → Silero VAD → Whisper `base` → Groq `llama-3.1-8b-instant` → Edge TTS

Per-turn times are in `voice_pipeline_results.csv` (end-to-end often ~25–64 seconds; TTS is usually the slow piece).

I also ran gTTS vs Coqui in `tts_evaluation/` on the same ten government-style sentences. gTTS was faster (~0.36 s mean vs ~1.88 s) and easier to set up, so the English loop in `voice_assistant/` uses gTTS instead of Edge TTS. Numbers: [`../tts_evaluation/TTS_COMPARISON.md`](../tts_evaluation/TTS_COMPARISON.md).

---

## Noise on English WER

Read-aloud WER and LibriSpeech are fairly clean. For a harder check I used `voice_assistant/english/asr_baseline/evaluate_asr_wer.py`: take in-scope public-service sentences, synthesize them so the text is known, mix white noise, run Whisper `small`. I tried 20 / 15 / 10 dB together, then kept **15 dB**.

Reported run: **60 utterances**, mean WER **14.18%** at 15 dB. CSV: `voice_assistant/outputs/english/asr_baseline_reports/asr_wer_results_20260601_174812.csv`.

---

## Dzongkha

Whisper with `language="dz"` (`DZwer.py`) was a first poke. Dzongkha is not Tibetan: configs have to say `dzo` / `dz`, never `bod` / `bo`. For ASR I used Meta MMS, which has a Dzongkha adapter.

`DZ_asr.py` is record → Silero VAD → MMS through the Hugging Face inference API. `DZ_tts.py` goes the other way: Google Translate + Gemini + `facebook/mms-tts-dzo`.

I also ran MMS locally (`facebook/mms-1b-all`, adapter `dzo`) on `Jyoti-77/dzongkha-asr`. Smoke test, mic/file demo, then a **10-clip** WER/CER run. At **15 dB**: mean WER **28.5%**, mean CER **16.8%**. WER needs splitting on tsheg (`་`) or JiWER treats a line as one word. `human_eval_asr.csv` is the listening sheet for a native speaker.

The demo’s bilingual path used Bhutan GovTech’s ASR and translation APIs. This folder is the MMS path I built.
