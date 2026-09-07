# English TTS evaluation (gTTS vs Coqui)

I compared two English engines on the same sentences. `English_tts.py` already spoke with Edge TTS; this folder is how I chose gTTS for the loop in `voice_assistant/`. Write-up: [`TTS_COMPARISON.md`](TTS_COMPARISON.md).

From repo root:

```bash
python tts_evaluation/gtts/run_gtts_demo.py --text "Your registration is completed."
python tts_evaluation/gtts/evaluate_gtts.py --repeats 1 --whisper-model tiny
python tts_evaluation/coqui_tts/run_coqui_demo.py --text "Your registration is completed."
python tts_evaluation/coqui_tts/evaluate_coqui.py --repeats 1 --whisper-model tiny
```

Audio lands under `outputs/` locally (gitignored). Reports stay in `*/outputs/*_eval_reports/`. Coqui is the offline backup. Dzongkha TTS in `voice_module/` is MMS-TTS, not these two.
