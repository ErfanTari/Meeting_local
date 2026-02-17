"""
Health monitoring and automatic recovery for the meeting pipeline.
"""
import os
import logging
import shutil
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

LM_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")


def check_lmstudio_alive(timeout: int = 3) -> bool:
    """Check if LM Studio is responding."""
    try:
        r = requests.get(f"{LM_URL}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def check_disk_space(path: Path = None, min_mb: int = 500) -> bool:
    """Check if there's enough disk space."""
    target = str(path) if path else "/"
    usage = shutil.disk_usage(target)
    free_mb = usage.free / (1024 * 1024)
    if free_mb < min_mb:
        logger.warning("Low disk space: %.0f MB free (minimum: %d MB)", free_mb, min_mb)
        return False
    return True


def check_blackhole_device() -> bool:
    """Check if BlackHole audio device is available."""
    import subprocess
    try:
        p = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5
        )
        output = (p.stdout or "") + "\n" + (p.stderr or "")
        return "BlackHole" in output
    except Exception:
        return False


class HealthMonitor:
    """Monitors pipeline health and provides recovery suggestions."""

    def __init__(self):
        self._lmstudio_was_down = False
        self._consecutive_capture_errors = 0
        self._consecutive_transcribe_errors = 0

    def on_capture_success(self):
        self._consecutive_capture_errors = 0

    def on_capture_error(self, error: Exception) -> str:
        """Returns recovery action: 'retry', 'backoff', or 'skip'."""
        self._consecutive_capture_errors += 1
        if self._consecutive_capture_errors <= 2:
            return "retry"
        elif self._consecutive_capture_errors <= 5:
            logger.warning("Multiple capture errors (%d), backing off", self._consecutive_capture_errors)
            return "backoff"
        else:
            logger.error("Too many capture errors (%d), skipping", self._consecutive_capture_errors)
            return "skip"

    def on_transcribe_success(self):
        self._consecutive_transcribe_errors = 0

    def on_transcribe_error(self, error: Exception) -> str:
        self._consecutive_transcribe_errors += 1
        if self._consecutive_transcribe_errors <= 3:
            return "retry"
        return "skip"

    def on_llm_error(self, error: Exception) -> str:
        """Returns 'retry' or 'transcribe_only'."""
        if not check_lmstudio_alive():
            if not self._lmstudio_was_down:
                logger.warning("LM Studio is down — switching to transcription-only mode")
                self._lmstudio_was_down = True
            return "transcribe_only"
        return "retry"

    def on_llm_success(self):
        if self._lmstudio_was_down:
            logger.info("LM Studio is back — resuming translation")
            self._lmstudio_was_down = False

    @property
    def is_lmstudio_down(self) -> bool:
        return self._lmstudio_was_down

    def run_preflight(self) -> dict:
        """Run all health checks and return results."""
        results = {
            "lmstudio": check_lmstudio_alive(),
            "disk_space": check_disk_space(),
            "blackhole": check_blackhole_device(),
        }

        for check, ok in results.items():
            if ok:
                logger.info("Preflight %s: OK", check)
            else:
                logger.warning("Preflight %s: FAILED", check)

        return results
