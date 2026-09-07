# English spoken loop

Mic (Silero VAD) → Whisper `small` → Groq `llama-3.1-8b-instant` → gTTS.

Same Whisper size as the noisy WER script in `../asr_baseline/`. WER is not computed during a live turn.

```bash
python voice_assistant/english/pipeline/run_text_demo.py --text "Where is permit office in Thimphu?"
python voice_assistant/english/pipeline/run_voice_demo.py
```

Needs `GROQ_API_KEY` in repo-root `.env`. Speak after the prompt; pause about a second when finished.

Outputs: `voice_assistant/outputs/english/pipeline_capture/` and `pipeline_replies/` (audio gitignored).

If OpenMP errors on Windows, see [`docs/CHALLENGES.md`](../../../docs/CHALLENGES.md).
