"""Shared configuration constants. All env var reads happen here."""
import os


def _bool_env(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes")


# LM Studio
LM_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LM_MODEL_FAST = os.getenv("LMSTUDIO_MODEL_FAST", "google/gemma-3-4b")
LM_MODEL_SMART = os.getenv("LMSTUDIO_MODEL_SMART", "google/gemma-3-4b")

# Feature flags
STREAM_TRANSLATION = _bool_env("STREAM_TRANSLATION", "false")
SKIP_EMPTY_CHUNKS = _bool_env("SKIP_EMPTY_CHUNKS", "true")
CLEANUP_WAV = _bool_env("CLEANUP_WAV", "true")

# Minutes
MINUTES_WINDOW = int(os.getenv("MINUTES_WINDOW", "600"))
