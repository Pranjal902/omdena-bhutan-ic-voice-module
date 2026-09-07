# Dzongkha ASR (local MMS)

Mic or file → Silero VAD → Meta MMS (`facebook/mms-1b-all`, adapter `dzo`) → Dzongkha text.

Whisper is not a Dzongkha model (`dzo`, not Tibetan `bod`). Public speech data is scarce, which is why this path is Meta MMS rather than copying the English Whisper stack.

`voice_module/voice_pipelines/Dzongkha/DZ_asr.py` uses the Hugging Face MMS API. This folder runs the same model locally.

Offline scores use `Jyoti-77/dzongkha-asr`. Listening sheet: `../eval/human_eval_asr.csv` and [`../eval/LIVE_MIC_TEST.md`](../eval/LIVE_MIC_TEST.md).

```bash
python voice_assistant/dzongkha/asr/run_mms_smoke_test.py
python voice_assistant/dzongkha/asr/run_dzongkha_transcribe_demo.py
python voice_assistant/dzongkha/asr/evaluate_mms_wer.py --limit 10 --include-clean
```

**10 clips, 15 dB SNR:** mean WER **28.5%**, mean CER **16.8%**. Report: `outputs/dzongkha/asr/dzongkha_mms_wer_results_20260612_230205.csv`.

WER tokenization splits on tsheg (`་`) in `mms_asr.py`. First run downloads the model (~1 GB). Optional `HF_TOKEN` in `.env` for Hugging Face.
