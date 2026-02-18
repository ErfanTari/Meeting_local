"""
Rich UI pipeline entry point.
Runs the same pipeline as live_loop_sys_with_minutes.py but with a rich terminal UI.

Usage:
    RICH_UI=true PYTHONPATH=. .venv/bin/python scripts/live_ui.py
"""
import logging
import os
from pathlib import Path

# Enable rich UI
os.environ.setdefault("RICH_UI", "true")

from app.pipeline import Pipeline

_log_dir = Path("out")
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_dir / "meeting_local.log", encoding="utf-8"),
    ],
)


def main():
    pipeline = Pipeline()
    pipeline.start()


if __name__ == "__main__":
    main()
