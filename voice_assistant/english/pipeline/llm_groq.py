"""Groq LLM helper for assistant responses."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Repo root: Voice-First (three levels above pipeline/)
REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

SYSTEM_PROMPT = """
You are a helpful Bhutan public-service assistant.
Answer in clear, short English.
Focus on permits, health services, business registration, office locations, and application status.
If unsure, ask the user to clarify.
""".strip()


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file before running the LLM pipeline."
        )
    return Groq(api_key=api_key)


def generate_reply(user_text: str, model: str = "llama-3.1-8b-instant") -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
