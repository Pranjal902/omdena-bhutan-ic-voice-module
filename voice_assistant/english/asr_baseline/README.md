# English noisy WER

Whisper `small` on public-service sentences from `shared/data/bhutan_classifier_test_set.csv` (in-scope, safe). Speech is synthesized so the text is known, then white noise is mixed in.

Pass/fail: mean WER under 20% at 15 dB SNR. I tried 20 / 15 / 10 dB first; 15 dB is the one I kept.

Reported full run (60 synthesized utterances at 15 dB): **14.18%** mean WER. CSV: `outputs/english/asr_baseline_reports/asr_wer_results_20260601_174812.csv`.

```bash
python voice_assistant/english/asr_baseline/evaluate_asr_wer.py --limit 10 --whisper-model small
python voice_assistant/english/asr_baseline/evaluate_asr_wer.py --whisper-model small
python voice_assistant/english/asr_baseline/evaluate_asr_wer.py --limit 10 --explore-snr --whisper-model small
```

`--include-clean` adds a no-noise reference. It is not the gate.
