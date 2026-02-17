import os
from datetime import datetime
from pathlib import Path

from app.ffmpeg_capture import record_chunk_avfoundation
from app.transcribe import Transcriber
from app.lmstudio_client import translate

SYSTEM_IDX = int(os.getenv("SYSTEM_AUDIO_IDX", "2"))
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "3"))
TARGET_LANG = os.getenv("TARGET_LANG", "English")

DATA_DIR = Path("data/live")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

transcriber = Transcriber(model_name=os.getenv("WHISPER_MODEL", "small"))

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_line(path: Path, line: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    print(f"SYS-only live loop running. idx={SYSTEM_IDX}, chunk={CHUNK_SECONDS}s. Ctrl+C to stop.")
    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys_wav = DATA_DIR / f"sys_{stamp}.wav"

        print(f"[{ts()}] recording...")
        try:
            record_chunk_avfoundation(SYSTEM_IDX, CHUNK_SECONDS, sys_wav)
        except Exception as e:
            print(f"[{ts()}] [CAPTURE ERROR] {e}")
            continue

        print(f"[{ts()}] transcribing...")
        sys_text = transcriber.transcribe_file(sys_wav)
        if not sys_text:
            print(f"[{ts()}] (empty transcript)")
            continue

        line = f"[{ts()}] [SYS] {sys_text}"
        print(line)
        append_line(OUT_DIR / "transcript.txt", line)

        print(f"[{ts()}] translating...")
        try:
            tr = translate(sys_text, target_lang=TARGET_LANG)
            line_tr = f"[{ts()}] [SYS->{TARGET_LANG}] {tr}"
            print(line_tr)
            append_line(OUT_DIR / "translation.txt", line_tr)
        except Exception as e:
            print(f"[{ts()}] [LLM ERROR] {e}")

if __name__ == "__main__":
    main()
