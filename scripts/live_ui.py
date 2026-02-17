"""
Rich UI pipeline entry point.
Runs the same pipeline as live_loop_sys_with_minutes.py but with a rich terminal UI.

Usage:
    RICH_UI=true PYTHONPATH=. .venv/bin/python scripts/live_ui.py
"""
import logging
import os

# Enable rich UI
os.environ.setdefault("RICH_UI", "true")

from app.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    pipeline = Pipeline()
    pipeline.start()


if __name__ == "__main__":
    main()
