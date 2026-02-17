import os
import logging
import signal
import time
import threading
import queue
import wave
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque

from app.ffmpeg_capture import record_chunk_avfoundation
from app.transcribe import Transcriber
from app.lmstudio_client import translate, summarize_block
from app.health import HealthMonitor
from app.ui import create_ui
from app.output import StructuredOutput

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        system_audio_idx: int = None,
        chunk_seconds: int = None,
        target_lang: str = None,
        summary_every_seconds: int = None,
        whisper_model: str = None,
        whisper_backend: str = None,
        vad_enabled: bool = None,
        data_dir: Path = None,
        out_dir: Path = None,
    ):
        self.system_audio_idx = system_audio_idx or int(os.getenv("SYSTEM_AUDIO_IDX", "2"))
        self.chunk_seconds = chunk_seconds or int(os.getenv("CHUNK_SECONDS", "10"))
        self.target_lang = target_lang or os.getenv("TARGET_LANG", "English")
        self.summary_every_seconds = summary_every_seconds or int(os.getenv("SUMMARY_EVERY_SECONDS", "300"))
        self.whisper_model = whisper_model or os.getenv("WHISPER_MODEL", "small")
        self.whisper_backend = whisper_backend or os.getenv("WHISPER_BACKEND", None)

        if vad_enabled is not None:
            self.vad_enabled = vad_enabled
        else:
            self.vad_enabled = os.getenv("VAD_ENABLED", "true").lower() in ("true", "1", "yes")

        self._vad = None

        self.data_dir = data_dir or Path("data/live")
        self.out_dir = out_dir or Path("out")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(exist_ok=True)

        self._stop_event = threading.Event()
        self._capture_queue = queue.Queue(maxsize=4)
        self._transcribe_queue = queue.Queue(maxsize=4)

        self._translation_buffer = deque(maxlen=1200)
        self._buffer_lock = threading.Lock()

        self._last_summary_text = ""

        self._transcriber = None
        self._health = HealthMonitor()
        self._meeting_start = datetime.now()
        self._ui = create_ui(meeting_start=self._meeting_start)
        self._structured_output = StructuredOutput(self.out_dir, meeting_start=self._meeting_start)

    def _ts(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_line(self, path: Path, line: str):
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _init_vad(self):
        if self.vad_enabled:
            try:
                from app.vad import has_speech, segment_audio, extract_segment_audio
                self._vad = {
                    "has_speech": has_speech,
                    "segment_audio": segment_audio,
                    "extract_segment": extract_segment_audio,
                }
                logger.info("VAD enabled (Silero)")
            except ImportError:
                logger.warning("VAD requested but silero-vad/torch not installed, falling back to fixed chunking")
                self.vad_enabled = False

    def _write_segment_wav(self, audio_data: np.ndarray, out_path: Path, sample_rate: int = 16000):
        """Write a numpy int16 array as a WAV file."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

    def _capture_worker(self):
        logger.info("Capture thread started (idx=%d, chunk=%ds, vad=%s)",
                     self.system_audio_idx, self.chunk_seconds, self.vad_enabled)
        self._init_vad()

        while not self._stop_event.is_set():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            wav_path = self.data_dir / f"sys_{stamp}.wav"
            try:
                record_chunk_avfoundation(self.system_audio_idx, self.chunk_seconds, wav_path)
                self._health.on_capture_success()
            except Exception as e:
                action = self._health.on_capture_error(e)
                logger.error("[CAPTURE ERROR] %s (action=%s)", e, action)
                if action == "backoff" and not self._stop_event.is_set():
                    time.sleep(3)
                elif action == "skip":
                    time.sleep(5)
                elif not self._stop_event.is_set():
                    time.sleep(1)
                continue

            if self.vad_enabled and self._vad:
                # Check for speech first — skip silent chunks entirely
                if not self._vad["has_speech"](wav_path):
                    logger.debug("No speech detected in %s, skipping", wav_path.name)
                    continue

                # Segment based on VAD boundaries
                segments = self._vad["segment_audio"](
                    wav_path,
                    min_chunk_seconds=3.0,
                    max_chunk_seconds=15.0,
                    silence_gap_seconds=0.5,
                )

                if not segments:
                    self._capture_queue.put(wav_path, timeout=10)
                    continue

                for i, (start_sec, end_sec) in enumerate(segments):
                    seg_audio = self._vad["extract_segment"](wav_path, start_sec, end_sec)
                    seg_path = self.data_dir / f"sys_{stamp}_seg{i}.wav"
                    self._write_segment_wav(seg_audio, seg_path)
                    self._capture_queue.put(seg_path, timeout=10)
            else:
                self._capture_queue.put(wav_path, timeout=10)

    def _transcribe_worker(self):
        logger.info("Transcription thread started (model=%s)", self.whisper_model)
        self._transcriber = Transcriber(
            model_name=self.whisper_model,
            backend=self.whisper_backend,
        )
        while not self._stop_event.is_set():
            try:
                wav_path = self._capture_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                text = self._transcriber.transcribe_file(wav_path)
            except Exception as e:
                logger.error("[TRANSCRIBE ERROR] %s", e)
                continue

            if not text:
                logger.debug("Empty transcript for %s", wav_path.name)
                continue

            self._ui.on_transcript(text)
            line = f"[{self._ts()}] [SYS] {text}"
            self._append_line(self.out_dir / "transcript.txt", line)

            self._transcribe_queue.put((text, self._ts()))

    def _translate_worker(self):
        logger.info("Translation thread started (target=%s)", self.target_lang)
        while not self._stop_event.is_set():
            try:
                text, timestamp = self._transcribe_queue.get(timeout=1)
            except queue.Empty:
                continue

            # If LM Studio is down, skip translation (transcription-only mode)
            if self._health.is_lmstudio_down:
                # Periodically re-check if LM Studio came back
                from app.health import check_lmstudio_alive
                if check_lmstudio_alive():
                    self._health.on_llm_success()
                else:
                    continue

            try:
                tr = translate(text, target_lang=self.target_lang).strip()
                self._health.on_llm_success()
            except Exception as e:
                action = self._health.on_llm_error(e)
                logger.error("[LLM ERROR] %s (action=%s)", e, action)
                if action == "transcribe_only":
                    continue
                # retry — item is already consumed, just skip this one
                continue

            if tr:
                self._ui.on_translation(tr, self.target_lang)
                line_tr = f"[{self._ts()}] [SYS->{self.target_lang}] {tr}"
                self._append_line(self.out_dir / "translation.txt", line_tr)

                self._structured_output.add_entry(text, translation=tr)

                with self._buffer_lock:
                    self._translation_buffer.append(f"[{self._ts()}] {tr}")

    def _minutes_worker(self):
        logger.info("Minutes thread started (every %ds)", self.summary_every_seconds)
        last_summary_time = time.time()
        last_buffer_idx = 0  # track how much of the buffer we've already summarized

        while not self._stop_event.is_set():
            elapsed = time.time() - last_summary_time
            if elapsed < self.summary_every_seconds:
                self._stop_event.wait(timeout=min(10, self.summary_every_seconds - elapsed))
                continue

            with self._buffer_lock:
                all_lines = list(self._translation_buffer)

            # Only summarize new lines since last summary
            new_lines = all_lines[last_buffer_idx:]
            if not new_lines:
                last_summary_time = time.time()
                continue

            block = "\n".join(new_lines)
            if not block.strip():
                last_summary_time = time.time()
                continue

            try:
                minutes = summarize_block(block, previous_summary=self._last_summary_text)
                self._last_summary_text = minutes
                last_buffer_idx = len(all_lines)

                md = self.out_dir / "rolling_minutes.md"
                txt = self.out_dir / "rolling_minutes.txt"
                md.write_text(f"# Rolling Minutes (updated {self._ts()})\n\n{minutes}\n", encoding="utf-8")
                txt.write_text(minutes + "\n", encoding="utf-8")
                self._ui.on_minutes_updated()
                logger.info("[MINUTES UPDATED] -> out/rolling_minutes.md")
            except Exception as e:
                logger.error("[MINUTES ERROR] %s", e)

            last_summary_time = time.time()

    def start(self):
        logger.info("Pipeline starting: idx=%d, chunk=%ds, target=%s, summary_every=%ds, vad=%s",
                     self.system_audio_idx, self.chunk_seconds, self.target_lang,
                     self.summary_every_seconds, self.vad_enabled)
        self._ui.on_status(f"Pipeline starting (idx={self.system_audio_idx}, chunk={self.chunk_seconds}s)")

        self._ui.start()

        threads = [
            threading.Thread(target=self._capture_worker, name="capture", daemon=True),
            threading.Thread(target=self._transcribe_worker, name="transcribe", daemon=True),
            threading.Thread(target=self._translate_worker, name="translate", daemon=True),
            threading.Thread(target=self._minutes_worker, name="minutes", daemon=True),
        ]

        for t in threads:
            t.start()

        def _shutdown(signum, frame):
            self._ui.on_status("Shutting down...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Block main thread until stop is requested
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=1)

        # Wait for threads to finish
        for t in threads:
            t.join(timeout=5)

        self._ui.stop()
        logger.info("Pipeline stopped.")

    def stop(self):
        self._stop_event.set()
