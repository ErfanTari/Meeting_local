import os
import time
from datetime import datetime
from pathlib import Path

from app.ffmpeg_capture import record_chunk_avfoundation
from app.transcribe import Transcriber
from app.lmstudio_client import translate

SYSTEM_IDX = int(os.getenv("SYSTEM_AUDIO_IDX", "2"))  # BlackHole
MIC_IDX = int(os.getenv("MIC_AUDIO_IDX", "0"))        # Mic

CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "15"))  # start small for low latency
TARGET_LANG = os.getenv("TARGET_LANG", "English")

DATA_DIR = Path("data/live")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

transcriber = Transcriber(model_name=os.getenv("WHISPER_MODEL", "small"))

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_line(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def one_cycle():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sys_wav = DATA_DIR / f"sys_{stamp}.wav"
    mic_wav = DATA_DIR / f"mic_{stamp}.wav"

    # record system then mic (simple v1). We'll parallelize later.
    record_chunk_avfoundation(SYSTEM_IDX, CHUNK_SECONDS, sys_wav)
    record_chunk_avfoundation(MIC_IDX, CHUNK_SECONDS, mic_wav)

    sys_text = transcriber.transcribe_file(sys_wav)
    mic_text = transcriber.transcribe_file(mic_wav)

    if sys_text:
        tr = translate(sys_text, target_lang=TARGET_LANG)
        line = f"[{now_ts()}] [SYS] {sys_text}"
        line_tr = f"[{now_ts()}] [SYS->{TARGET_LANG}] {tr}"
        print(line)
        print(line_tr)
        append_line(OUT_DIR / "transcript.txt", line)
        append_line(OUT_DIR / "translation.txt", line_tr)

    if mic_text:
        tr = translate(mic_text, target_lang=TARGET_LANG)
        line = f"[{now_ts()}] [MIC] {mic_text}"
        line_tr = f"[{now_ts()}] [MIC->{TARGET_LANG}] {tr}"
        print(line)
        print(line_tr)
        append_line(OUT_DIR / "transcript.txt", line)
        append_line(OUT_DIR / "translation.txt", line_tr)

def main():
    print("Live loop running. Stop with Ctrl+C.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        t0 = time.time()
        one_cycle()
        # no sleep needed; recording time is the pacing

if __name__ == "__main__":
    main()
