import json
import logging
import requests
from typing import List, Dict, Any, Generator

from app.config import LM_URL, LM_MODEL_FAST, LM_MODEL_SMART

logger = logging.getLogger(__name__)


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


def chat_stream(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 120) -> Generator[str, None, None]:
    """Streaming chat — yields partial tokens as they arrive."""
    url = f"{LM_URL}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    with requests.post(url, json=payload, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


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


def translate_stream(text: str, target_lang: str = "English") -> Generator[str, None, None]:
    """Streaming translation — yields partial tokens."""
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
    yield from chat_stream(LM_MODEL_FAST, messages, temperature=0.0, timeout=60)


def summarize_block(transcript_block: str, previous_summary: str = "") -> str:
    """Summarize transcript. Uses rolling window if previous_summary is provided."""
    if previous_summary:
        user_content = (
            f"Previous summary:\n{previous_summary}\n\n"
            f"New transcript since last summary:\n{transcript_block}"
        )
        system_content = (
            "You are a meeting assistant.\n"
            "You have a previous summary and new transcript text.\n"
            "Update the summary to incorporate the new transcript.\n"
            "Output Markdown with sections:\n"
            "## Summary\n## Decisions\n## Action Items\n## Open Questions\n"
            "If a section has none, write 'None'.\n"
            "Do not invent.\n"
        )
    else:
        user_content = transcript_block
        system_content = (
            "You are a meeting assistant.\n"
            "Summarize ONLY what is in the transcript.\n"
            "Output Markdown with sections:\n"
            "## Summary\n## Decisions\n## Action Items\n## Open Questions\n"
            "If a section has none, write 'None'.\n"
            "Do not invent.\n"
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return chat(LM_MODEL_SMART, messages, temperature=0.2, timeout=180)
