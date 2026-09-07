# gTTS vs Coqui TTS — English evaluation

I ran both engines on the same ten government-style English sentences. `English_tts.py` already used Edge TTS; this comparison is how I picked gTTS for the English loop in `voice_assistant/`.

| Metric | gTTS | Coqui TTS |
|---|---:|---:|
| Success rate | 100% | 100% |
| Mean latency | ~0.36 sec | ~1.88 sec |
| P95 latency | ~0.64 sec | ~5.19 sec |
| Mean proxy WER | ~0.047 | ~0.097 |

Reports:

- `gtts/outputs/gtts_eval_reports/gtts_eval_results_20260528_224557.csv`
- `coqui_tts/outputs/coqui_eval_reports/coqui_eval_results_20260528_225803.csv`

```bash
python tts_evaluation/gtts/evaluate_gtts.py --repeats 1 --whisper-model tiny
python tts_evaluation/coqui_tts/evaluate_coqui.py --repeats 1 --whisper-model tiny
```

Proxy WER only checks whether Whisper can read the audio back. `human_eval_template.csv` in each engine folder is for listening scores. Dzongkha TTS in `voice_module/` is MMS-TTS, not these two.
