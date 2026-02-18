"""
Threaded pipeline entry point: capture -> transcribe -> translate -> minutes.
All configuration via environment variables (see CLAUDE.md).

Usage:
    python scripts/live_loop_sys_with_minutes.py           # normal mode
    python scripts/live_loop_sys_with_minutes.py --dry-run  # test with sample audio
"""
import sys
import logging
from pathlib import Path
from app.pipeline import Pipeline
from app.health import HealthMonitor

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


def dry_run():
    """Run preflight checks and test the pipeline with a sample if available."""
    print("=== Dry Run Mode ===\n")

    monitor = HealthMonitor()
    results = monitor.run_preflight()

    all_ok = all(results.values())
    if not all_ok:
        print("\nSome preflight checks failed. See warnings above.")
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
    else:
        print("\nAll preflight checks passed.")

    # Try to transcribe a sample file if available
    from pathlib import Path
    sample_files = list(Path("data").glob("*.wav"))
    if sample_files:
        print(f"\nTesting transcription with {sample_files[0]}...")
        from app.transcribe import Transcriber
        import time
        t = Transcriber()
        t0 = time.time()
        text = t.transcribe_file(sample_files[0])
        elapsed = time.time() - t0
        print(f"  Backend: {t.backend}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Result: {text[:100] if text else '(empty)'}...")
    else:
        print("\nNo sample WAV files found in data/ for transcription test.")

    print("\nDry run complete.")


def main():
    if "--dry-run" in sys.argv:
        dry_run()
        return

    pipeline = Pipeline()
    pipeline.start()


if __name__ == "__main__":
    main()
