# Building Voice-First AI Solutions for Public Services — Voice Module

My voice work from **[Building Voice-First AI Solutions for Public Services](https://www.omdena.com/projects/building-voice-first-ai-solutions-for-public-services)**, an 8-week Omdena Innovation Challenge (May–July 2026), where I worked as an ML Engineer on the Voice Module.

## The problem

Public-service information in Bhutan is easier to miss if you cannot comfortably use text portals. The challenge asked whether a citizen could speak a question — English or Dzongkha — and hear an answer. My team owned turning speech into text, turning answers back into speech, and checking accuracy and latency.

Dzongkha is low-resource: little public text or speech for public-service work, and Whisper is not a Dzongkha ASR model. The demo’s bilingual path used Bhutan GovTech’s ASR and translation APIs. The Dzongkha work in this repo is Meta MMS.

## My role

I worked on the voice I/O side of the challenge: speech in, speech out, and whether that path was accurate and fast enough. That covered English capture and transcription, word-error checks, a spoken English loop, an English TTS comparison, and Dzongkha ASR and TTS with Meta MMS.

## Where this fit into the larger project

The challenge ran four tracks in parallel.

| Team | Focus |
|---|---|
| Team A | Multilingual NLP / LLM (intent, conversational layer, RAG) |
| Team B | Agentic workflows (LangGraph routing, tools, safety) |
| Team C | Voice Module *(mine)* — ASR, VAD, TTS, latency, WER |
| Team D | Integration, traces, APIs, docs |

| Team | Connection to my work |
|---|---|
| Team A — NLP | Took transcripts (English, or Dzongkha after translation) as input |
| Team B — Agentic | Routed on understood text |
| Team D — Integration | Traces, APIs, and integration glue around the shared MVP |

Here's a [project demo](https://drive.google.com/file/d/1OTeVDa7y4xWdxCfFhXL43-pELMyHVc4Z/view?usp=sharing).

Here's the [final project documentation](https://docs.google.com/document/d/12GBB1sQClfnQfCocxG04o2Yjj5QjAwUi/edit?usp=sharing), prepared by our Omdena project lead.

## What's in this repo

| Folder | What it is |
|---|---|
| [`voice_module/`](voice_module) | English ASR + VAD + latency, mic WER, Edge TTS + Groq loop, LibriSpeech dummy baseline, Dzongkha MMS (HF API) and TTS scripts |
| [`tts_evaluation/`](tts_evaluation) | English gTTS vs Coqui. I used **gTTS** on the English loop in `voice_assistant/` |
| [`voice_assistant/`](voice_assistant) | English loop (Whisper `small` → Groq → gTTS), 15 dB WER on synthesized speech (**14.18%** mean on 60 sentences), local MMS Dzongkha ASR |
| [`docs/JOURNEY.md`](docs/JOURNEY.md) | How the voice work moved |
| [`docs/CHALLENGES.md`](docs/CHALLENGES.md) | What broke while I was building it |

Audio from eval runs is not in git. Result CSVs are.

## Setup

Python 3, from this repo root:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Add only the keys you need:

- `GROQ_API_KEY` — English Groq demos
- `HF_TOKEN` — Hugging Face MMS API and dataset download
- `GEMINI_API_KEY` and `GOOGLE_TRANSLATE_API_KEY` — only for `DZ_tts.py`

ffmpeg helps Whisper read generated audio (`imageio-ffmpeg` is in requirements). Windows OpenMP crash: [`docs/CHALLENGES.md`](docs/CHALLENGES.md).

## Run (from repo root)

English ASR + VAD — run these **inside** `voice_module/` so CSVs append next to the ones already there, then return to the repo root:

```bash
cd voice_module
python voice_pipelines/English/English_asr.py
python voice_pipelines/English/English_wer.py
python voice_pipelines/English/English_tts.py
cd ..
```

TTS comparison:

```bash
python tts_evaluation/gtts/run_gtts_demo.py --text "Your registration is completed."
python tts_evaluation/gtts/evaluate_gtts.py --repeats 1 --whisper-model tiny
```

English spoken loop and noisy WER:

```bash
python voice_assistant/english/pipeline/run_text_demo.py --text "Where is permit office in Thimphu?"
python voice_assistant/english/pipeline/run_voice_demo.py
python voice_assistant/english/asr_baseline/evaluate_asr_wer.py --limit 10 --whisper-model small
```

Dzongkha local MMS (first run downloads the model, ~1 GB):

```bash
python voice_assistant/dzongkha/asr/run_mms_smoke_test.py
python voice_assistant/dzongkha/asr/run_dzongkha_transcribe_demo.py
python voice_assistant/dzongkha/asr/evaluate_mms_wer.py --limit 10 --include-clean
```

More pipeline commands: [`voice_module/README.md`](voice_module/README.md).

## Skills this work draws on

- Mic capture and Silero VAD
- Whisper / faster-whisper and Meta MMS (`dzo`)
- WER (and Dzongkha CER) with a 15 dB noise check
- Groq for English replies; Edge TTS vs gTTS vs Coqui
