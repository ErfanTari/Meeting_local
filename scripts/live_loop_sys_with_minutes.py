import os
import time
from datetime import datetime
from pathlib import Path
from collections import deque

from app.ffmpeg_capture import record_chunk_avfoundation
from app.transcribe import Transcriber
from app.lmstudio_client import translate, summarize_block

SYSTEM_IDX = int(os.getenv("SYSTEM_AUDIO_IDX", "2"))

CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "10"))
OVERLAP_SECONDS = int(os.getenv("OVERLAP_SECONDS", "2"))
STEP_SLEEP = max(0, CHUNK_SECONDS - OVERLAP_SECONDS)

TARGET_LANG = os.getenv("TARGET_LANG", "English")
SUMMARY_EVERY_SECONDS = int(os.getenv("SUMMARY_EVERY_SECONDS", "300"))  # 5 min

DATA_DIR = Path("data/live")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

transcriber = Transcriber(model_name=os.getenv("WHISPER_MODEL", "small"))

# Keep recent lines (translations are what we summarize)
translation_buffer = deque(maxlen=1200)

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_line(path: Path, line: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_minutes():
    # Summarize the most recent portion of translated text
    block = "\n".join(list(translation_buffer)[-400:])  # cap prompt size
    if not block.strip():
        return

    minutes = summarize_block(block)

    md = OUT_DIR / "rolling_minutes.md"
    txt = OUT_DIR / "rolling_minutes.txt"

    md.write_text(f"# Rolling Minutes (updated {ts()})\n\n{minutes}\n", encoding="utf-8")
    txt.write_text(minutes + "\n", encoding="utf-8")

    print(f"\n[{ts()}] [MINUTES UPDATED] -> out/rolling_minutes.md\n")

def main():
    print(f"SYS loop + minutes. idx={SYSTEM_IDX}, chunk={CHUNK_SECONDS}s, overlap={OVERLAP_SECONDS}s, step={STEP_SLEEP}s")
    print("Ctrl+C to stop.\n")

    last_summary = time.time()

    while True:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys_wav = DATA_DIR / f"sys_{stamp}.wav"

        try:
            record_chunk_avfoundation(SYSTEM_IDX, CHUNK_SECONDS, sys_wav)
        except Exception as e:
            print(f"[{ts()}] [CAPTURE ERROR] {e}")
            time.sleep(1)
            continue

        sys_text = transcriber.transcribe_file(sys_wav)
        if not sys_text:
            # still advance time; no translation added
            time.sleep(STEP_SLEEP)
            continue

        # Live transcript
        line = f"[{ts()}] [SYS] {sys_text}"
        print(line)
        append_line(OUT_DIR / "transcript.txt", line)

        # Live translation
        try:
            tr = translate(sys_text, target_lang=TARGET_LANG).strip()
        except Exception as e:
            print(f"[{ts()}] [LLM ERROR] {e}")
            tr = ""

        if tr:
            line_tr = f"[{ts()}] [SYS->{TARGET_LANG}] {tr}"
            print(line_tr)
            append_line(OUT_DIR / "translation.txt", line_tr)

            # Buffer only the translated content for summarization
            translation_buffer.append(f"[{ts()}] {tr}")

        # Periodic minutes update
        if time.time() - last_summary >= SUMMARY_EVERY_SECONDS:
            try:
                write_minutes()
            except Exception as e:
                print(f"[{ts()}] [MINUTES ERROR] {e}")
            last_summary = time.time()

        time.sleep(STEP_SLEEP)

if __name__ == "__main__":
    main()
