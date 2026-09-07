# Voice assistant

English spoken loop, noisy English WER, and local Dzongkha MMS. Numbers and sequence: [`docs/JOURNEY.md`](../docs/JOURNEY.md).

| Path | What |
|------|------|
| [`english/pipeline/`](english/pipeline/README.md) | Mic → Whisper → Groq → gTTS |
| [`english/asr_baseline/`](english/asr_baseline/README.md) | 15 dB WER |
| [`dzongkha/asr/`](dzongkha/asr/README.md) | Local MMS `dzo` |
| [`dzongkha/eval/`](dzongkha/eval/LIVE_MIC_TEST.md) | Native-speaker mic note |
| `outputs/` | CSV reports (audio gitignored) |
