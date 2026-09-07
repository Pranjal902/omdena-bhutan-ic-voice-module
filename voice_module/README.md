# Voice module

# Voice module

English and Dzongkha pipelines. Run them from **this directory** so latency/WER CSVs append in place.

```
voice_module/
├── voice_pipelines/English/     English_asr.py, English_wer.py, English_tts.py
├── voice_pipelines/Dzongkha/    DZ_asr.py, DZ_tts.py, DZwer.py
├── asr/                         LibriSpeech dummy (faster-whisper tiny vs medium)
├── latency_results.csv
├── wer_results.csv
└── voice_pipeline_results.csv
```

From repo root, copy `.env.example` to `.env` first.

```bash
cd voice_module
python voice_pipelines/English/English_asr.py
python voice_pipelines/English/English_wer.py
python voice_pipelines/English/English_tts.py
python asr/asr_eng_baseline_metrics_dummyLibriSpeech.py
python voice_pipelines/Dzongkha/DZ_asr.py
```

`English_tts.py` needs `GROQ_API_KEY`. `DZ_asr.py` needs `HF_TOKEN`. `DZ_tts.py` needs Gemini + Google Translate keys.

The LibriSpeech script uses CUDA. Saved numbers: `asr/result_asr_eng_baseline_metrics_dummyLibriSpeech.txt` (tiny 11.21% WER, medium 6.60%).
