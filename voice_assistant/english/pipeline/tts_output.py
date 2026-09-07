"""gTTS output helper for assistant responses."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from gtts import gTTS


def speak_to_file(text: str, output_dir: str | Path) -> Path:
    if not text.strip():
        raise ValueError("Response text must not be empty.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out / f"assistant_reply_{timestamp}.mp3"
    gTTS(text=text, lang="en").save(str(output_path))
    return output_path
