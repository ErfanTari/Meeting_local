import subprocess
import time
from pathlib import Path

def record_chunk_avfoundation(audio_index: int, seconds: int, out_path: Path, sample_rate: int = 16000, retries: int = 3) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel", "error",
        "-f", "avfoundation",
        "-i", f":{audio_index}",
        "-t", str(seconds),
        "-ar", str(sample_rate),
        "-ac", "1",
        str(out_path),
    ]

    last_err = None
    # Give ffmpeg a hard timeout slightly above chunk length
    timeout_s = max(seconds + 4, 8)

    for attempt in range(1, retries + 1):
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
            if p.returncode == 0:
                return
            last_err = (p.stderr or "").strip()[-800:] or f"returncode={p.returncode}"
        except subprocess.TimeoutExpired:
            last_err = f"ffmpeg timed out after {timeout_s}s (device idx {audio_index})"
        time.sleep(0.3 * attempt)

    raise RuntimeError(f"ffmpeg capture failed (idx={audio_index}) after {retries} tries. Last error:\n{last_err}")
