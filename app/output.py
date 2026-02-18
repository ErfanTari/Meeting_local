"""
Structured output writers: JSON and SRT formats alongside the existing .txt/.md files.
Batched writes to avoid rewriting full files on every entry.
"""
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class TranscriptEntry:
    def __init__(self, text: str, translation: str = "", timestamp: datetime = None,
                 meeting_start: datetime = None):
        self.text = text
        self.translation = translation
        self.timestamp = timestamp or datetime.now()
        self.meeting_start = meeting_start or self.timestamp
        self._index = 0

    @property
    def relative_seconds(self) -> float:
        return (self.timestamp - self.meeting_start).total_seconds()

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "relative_seconds": self.relative_seconds,
            "text": self.text,
            "translation": self.translation,
        }


class StructuredOutput:
    """Manages structured output in JSON and SRT formats with batched writes."""

    def __init__(self, out_dir: Path, meeting_start: datetime = None,
                 flush_interval: float = 30.0):
        self.out_dir = out_dir
        self.meeting_start = meeting_start or datetime.now()
        self._entries: List[TranscriptEntry] = []
        self._counter = 0
        self._flush_interval = flush_interval
        self._dirty = False
        self._last_flush = time.time()
        self._lock = threading.Lock()

    def add_entry(self, text: str, translation: str = ""):
        with self._lock:
            self._counter += 1
            entry = TranscriptEntry(
                text=text,
                translation=translation,
                timestamp=datetime.now(),
                meeting_start=self.meeting_start,
            )
            entry._index = self._counter
            self._entries.append(entry)
            self._dirty = True

            # Flush if enough time has passed since last write
            now = time.time()
            if now - self._last_flush >= self._flush_interval:
                self._flush()

    def _flush(self):
        """Write JSON and SRT to disk. Must be called with self._lock held."""
        if not self._dirty:
            return
        self._write_json()
        self._write_srt()
        self._dirty = False
        self._last_flush = time.time()

    def flush_final(self):
        """Force flush at pipeline shutdown."""
        with self._lock:
            self._flush()

    def _write_json(self):
        data = {
            "meeting_start": self.meeting_start.isoformat(),
            "entries": [e.to_dict() for e in self._entries],
        }
        path = self.out_dir / "transcript.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_srt(self):
        lines = []
        for i, entry in enumerate(self._entries):
            start_sec = entry.relative_seconds
            # Estimate end time as start + 10s or until next entry
            if i + 1 < len(self._entries):
                end_sec = self._entries[i + 1].relative_seconds
            else:
                end_sec = start_sec + 10.0

            start_ts = _seconds_to_srt_time(start_sec)
            end_ts = _seconds_to_srt_time(end_sec)

            text = entry.translation if entry.translation else entry.text
            lines.append(f"{entry._index}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text)
            lines.append("")

        path = self.out_dir / "transcript.srt"
        path.write_text("\n".join(lines), encoding="utf-8")


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = timedelta(seconds=max(0, seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
