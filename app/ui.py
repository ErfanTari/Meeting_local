"""
Rich terminal UI for the meeting pipeline.
Shows live transcript + translation in a split view with a status bar.
"""
import os
import time
import threading
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

# Try to use rich for enhanced UI, fall back to plain text
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    logger.info("rich not installed, using plain text output")


class PlainUI:
    """Fallback UI that just prints to stdout (same as current behavior)."""

    def __init__(self, meeting_start: datetime = None):
        self._start = meeting_start or datetime.now()

    def _relative_ts(self):
        elapsed = datetime.now() - self._start
        total_seconds = int(elapsed.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def on_transcript(self, text: str):
        print(f"[{self._relative_ts()}] [SYS] {text}")

    def on_translation(self, text: str, target_lang: str):
        print(f"[{self._relative_ts()}] [SYS->{target_lang}] {text}")

    def on_minutes_updated(self):
        print(f"\n[{self._relative_ts()}] [MINUTES UPDATED] -> out/rolling_minutes.md\n")

    def on_status(self, msg: str):
        print(f"[{self._relative_ts()}] {msg}")

    def on_error(self, msg: str):
        print(f"[{self._relative_ts()}] [ERROR] {msg}")

    def start(self):
        pass

    def stop(self):
        pass


class RichUI:
    """Rich terminal UI with split transcript/translation view and status bar."""

    def __init__(self, meeting_start: datetime = None):
        self._start = meeting_start or datetime.now()
        self._console = Console()
        self._transcripts = deque(maxlen=50)
        self._translations = deque(maxlen=50)
        self._status_msg = "Starting..."
        self._error_msg = ""
        self._minutes_count = 0
        self._lock = threading.Lock()
        self._live = None
        self._running = False

    def _relative_ts(self):
        elapsed = datetime.now() - self._start
        total_seconds = int(elapsed.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def on_transcript(self, text: str):
        with self._lock:
            self._transcripts.append(f"[{self._relative_ts()}] {text}")

    def on_translation(self, text: str, target_lang: str):
        with self._lock:
            self._translations.append(f"[{self._relative_ts()}] {text}")

    def on_minutes_updated(self):
        with self._lock:
            self._minutes_count += 1
            self._status_msg = f"Minutes updated ({self._minutes_count}x)"

    def on_status(self, msg: str):
        with self._lock:
            self._status_msg = msg

    def on_error(self, msg: str):
        with self._lock:
            self._error_msg = msg

    def _render(self):
        layout = Layout()

        # Build transcript panel
        with self._lock:
            transcript_text = "\n".join(self._transcripts) if self._transcripts else "(waiting for audio...)"
            translation_text = "\n".join(self._translations) if self._translations else "(waiting for translation...)"
            status = self._status_msg
            error = self._error_msg

        layout.split_column(
            Layout(name="main", ratio=9),
            Layout(name="status", size=3),
        )

        layout["main"].split_row(
            Layout(Panel(transcript_text, title="Transcript", border_style="blue"), name="left"),
            Layout(Panel(translation_text, title="Translation", border_style="green"), name="right"),
        )

        status_parts = [f"Elapsed: {self._relative_ts()}"]
        if status:
            status_parts.append(status)
        if error:
            status_parts.append(f"[red]{error}[/red]")

        layout["status"].update(Panel(" | ".join(status_parts), border_style="yellow"))

        return layout

    def start(self):
        self._running = True
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=2,
            screen=True,
        )
        self._live.start()

        def _update_loop():
            while self._running:
                try:
                    self._live.update(self._render())
                except Exception:
                    pass
                time.sleep(0.5)

        self._update_thread = threading.Thread(target=_update_loop, name="ui-update", daemon=True)
        self._update_thread.start()

    def stop(self):
        self._running = False
        if self._live:
            self._live.stop()


def create_ui(rich_enabled: bool = None, meeting_start: datetime = None):
    """Factory function to create the appropriate UI."""
    if rich_enabled is None:
        rich_enabled = os.getenv("RICH_UI", "false").lower() in ("true", "1", "yes")

    if rich_enabled and HAS_RICH:
        return RichUI(meeting_start)
    return PlainUI(meeting_start)
