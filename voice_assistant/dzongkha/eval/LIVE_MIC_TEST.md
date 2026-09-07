# Live mic test — for Bhutanese reviewers

Speak Dzongkha into the mic and read what MMS prints.

```bash
python voice_assistant/dzongkha/asr/run_dzongkha_transcribe_demo.py
```

Wait for **Speak now...**, say a sentence, pause. First run downloads MMS (~1 GB).

Windows OpenMP error:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python voice_assistant/dzongkha/asr/run_dzongkha_transcribe_demo.py
```

Replay a file: add `--audio path/to/clip.wav`.

Saved files (local only): `outputs/dzongkha/asr/mic_capture/` and `transcripts/`. Offline sheet: [`human_eval_asr.csv`](human_eval_asr.csv).
