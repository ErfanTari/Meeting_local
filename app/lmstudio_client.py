import os
import requests
from typing import List, Dict, Any

# Base URL for LM Studio OpenAI-compatible server
LM_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")

# Models (set via env vars; defaults are safe)
LM_MODEL_FAST = os.getenv("LMSTUDIO_MODEL_FAST", "google/gemma-3-4b")
LM_MODEL_SMART = os.getenv("LMSTUDIO_MODEL_SMART", "google/gemma-3-4b")

def chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 120) -> str:
    url = f"{LM_URL}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def translate(text: str, target_lang: str = "English") -> str:
    messages = [
        {
            "role": "system",
            "content": (
                f"Translate the user text to {target_lang}.\n"
                "Rules:\n"
                "- Output ONLY the translation.\n"
                "- No explanations, no notes, no options, no extra lines.\n"
                "- Preserve meaning and tone.\n"
            ),
        },
        {"role": "user", "content": text},
    ]
    return chat(LM_MODEL_FAST, messages, temperature=0.0, timeout=60)

def summarize_block(transcript_block: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meeting assistant.\n"
                "Summarize ONLY what is in the transcript.\n"
                "Output Markdown with sections:\n"
                "## Summary\n## Decisions\n## Action Items\n## Open Questions\n"
                "If a section has none, write 'None'.\n"
                "Do not invent.\n"
            ),
        },
        {"role": "user", "content": transcript_block},
    ]
    return chat(LM_MODEL_SMART, messages, temperature=0.2, timeout=180)
