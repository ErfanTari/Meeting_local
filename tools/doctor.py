import os
import sys
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)

LM_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
MODEL_FAST = os.getenv("LMSTUDIO_MODEL_FAST", "google/gemma-3-4b")
MODEL_SMART = os.getenv("LMSTUDIO_MODEL_SMART", "google/gemma-3-4b")

SYSTEM_IDX = int(os.getenv("SYSTEM_AUDIO_IDX", "2"))  # BlackHole 2ch in your list
MIC_IDX = int(os.getenv("MIC_AUDIO_IDX", "0"))        # Logitech headset in your list

def run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)

def ok(msg: str): print(f"✅ {msg}")
def warn(msg: str): print(f"⚠️  {msg}")
def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)

def check_python():
    ok(f"Python executable: {sys.executable}")
    ok(f"Python version: {sys.version.splitlines()[0]}")
    if "anaconda" in sys.version.lower():
        warn("This Python build looks like Anaconda. Not fatal, but can cause native-lib issues. Brew Python is cleaner.")

def check_ffmpeg():
    try:
        p = run(["ffmpeg", "-version"])
        ok("ffmpeg is installed")
    except Exception as e:
        fail("ffmpeg not found. Install: brew install ffmpeg")

def list_avfoundation_devices() -> str:
    # ffmpeg prints device list to stderr for this command
    p = run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""], check=False)
    return (p.stdout or "") + "\n" + (p.stderr or "")

def check_audio_devices():
    text = list_avfoundation_devices()
    if "AVFoundation audio devices:" not in text:
        fail("Could not list avfoundation audio devices. Something is wrong with ffmpeg/permissions.")
    ok("Can list avfoundation devices")

    # Check for BlackHole virtual audio device
    if "BlackHole" in text:
        ok("BlackHole virtual audio device detected")
    else:
        warn("BlackHole not found in device list. Install BlackHole for system audio capture.")

    if f"[{SYSTEM_IDX}]" not in text:
        warn(f"SYSTEM_AUDIO_IDX={SYSTEM_IDX} not found in device list. Update env or your device order changed.")
    else:
        ok(f"Found system audio device index {SYSTEM_IDX}")

    if f"[{MIC_IDX}]" not in text:
        warn(f"MIC_AUDIO_IDX={MIC_IDX} not found in device list. Update env or your device order changed.")
    else:
        ok(f"Found mic audio device index {MIC_IDX}")

def check_disk_space():
    import shutil
    usage = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = usage.free / (1024 ** 3)
    if free_gb < 1.0:
        warn(f"Low disk space: {free_gb:.1f} GB free")
    else:
        ok(f"Disk space: {free_gb:.1f} GB free")

def check_wavs_optional():
    sys_wav = DATA_DIR / "system.wav"
    mic_wav = DATA_DIR / "mic.wav"
    if sys_wav.exists():
        ok(f"Found {sys_wav}")
    else:
        warn(f"Missing {sys_wav} (ok if you haven't recorded yet)")
    if mic_wav.exists():
        ok(f"Found {mic_wav}")
    else:
        warn(f"Missing {mic_wav} (ok if you haven't recorded yet)")

def check_apple_silicon():
    is_arm = platform.machine() == "arm64" and platform.system() == "Darwin"
    if is_arm:
        ok("Apple Silicon detected (arm64)")
        try:
            import mlx_whisper  # noqa: F401
            ok("mlx-whisper is installed — recommended backend for Apple Silicon")
        except ImportError:
            warn("mlx-whisper not installed. Install with: pip install mlx-whisper")
            warn("MLX backend gives ~2-3x speedup on Apple Silicon vs faster-whisper")
    else:
        ok(f"Architecture: {platform.machine()} — using faster-whisper backend")

def check_whisper_optional():
    sys_wav = DATA_DIR / "system.wav"
    mic_wav = DATA_DIR / "mic.wav"
    if not sys_wav.exists() and not mic_wav.exists():
        warn("No WAVs found; skipping Whisper test.")
        return

    # Use the unified Transcriber which auto-detects backend
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from app.transcribe import Transcriber
        ok(f"Transcriber import OK")
    except Exception as e:
        fail(f"Transcriber import failed: {e}")

    transcriber = Transcriber(model_name="small")
    ok(f"Whisper model loaded (backend={transcriber.backend}, model=small)")

    def transcribe_and_bench(path: Path, label: str):
        t0 = time.time()
        text = transcriber.transcribe_file(path)
        elapsed = time.time() - t0
        if text:
            ok(f"Whisper transcript ({label}) in {elapsed:.1f}s: {text[:80]}...")
        else:
            warn(f"Whisper transcript for {label} is empty (maybe silent).")

    if sys_wav.exists():
        transcribe_and_bench(sys_wav, "system.wav")
    if mic_wav.exists():
        transcribe_and_bench(mic_wav, "mic.wav")

def check_lmstudio():
    import requests
    try:
        r = requests.get(f"{LM_URL}/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        ok(f"LM Studio reachable at {LM_URL}")
    except Exception as e:
        fail(f"LM Studio not reachable at {LM_URL}. Start server in LM Studio. Error: {e}")

    models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
    if not models:
        warn("LM Studio returned no models (?)")
    else:
        ok(f"LM Studio models count: {len(models)}")

    if MODEL_FAST in models:
        ok(f"Fast model found: {MODEL_FAST}")
    else:
        warn(f"Fast model not found: {MODEL_FAST}. Set LMSTUDIO_MODEL_FAST to one from /v1/models")

    if MODEL_SMART in models:
        ok(f"Smart model found: {MODEL_SMART}")
    else:
        warn(f"Smart model not found: {MODEL_SMART}. Set LMSTUDIO_MODEL_SMART to one from /v1/models")

def check_translation_sanity():
    # Optional: only if lmstudio_client exists
    try:
        # ensure project root on path
        sys.path.insert(0, str(PROJECT_ROOT))
        from app.lmstudio_client import chat, LMSTUDIO_MODEL_FAST
        ok("Imported app.lmstudio_client")
    except Exception as e:
        warn(f"Skipping translation sanity (import failed): {e}")
        return

    text_tr = "Bu bir test cümlesidir."
    msgs = [
        {"role": "system", "content": "Translate Turkish to English. Output ONLY the translation. No explanations."},
        {"role": "user", "content": text_tr}
    ]
    try:
        out = chat(os.getenv("LMSTUDIO_MODEL_FAST", LMSTUDIO_MODEL_FAST), msgs, temperature=0.0, timeout=30).strip()
        if "\n" in out or "Option" in out or "Breakdown" in out:
            warn(f"Translation output looks verbose; tighten prompt. Output was: {out[:120]}...")
        else:
            ok(f"Translation sanity OK: {out}")
    except Exception as e:
        warn(f"Translation sanity failed: {e}")

def main():
    print("=== meeting_local doctor ===")
    check_python()
    check_apple_silicon()
    check_ffmpeg()
    check_audio_devices()
    check_disk_space()
    check_wavs_optional()
    check_whisper_optional()
    check_lmstudio()
    check_translation_sanity()
    print("\nAll checks done.")

if __name__ == "__main__":
    main()
