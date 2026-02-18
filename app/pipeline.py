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

from app.config import CLEANUP_WAV, SKIP_EMPTY_CHUNKS, STREAM_TRANSLATION, MINUTES_WINDOW
from app.ffmpeg_capture import record_chunk_avfoundation
from app.transcribe import Transcriber
from app.lmstudio_client import translate, translate_stream, summarize_block
from app.health import HealthMonitor
from app.ui import create_ui
from app.output import StructuredOutput

logger = logging.getLogger(__name__)

# Common Whisper hallucinations on silent/near-silent audio
_HALLUCINATION_PATTERNS = frozenset({
    "thank you", "thanks for watching", "thanks for listening",
    "you", "bye", "the end", "thank you for watching",
    "subscribe", "like and subscribe",
})


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
        self._pause_event = threading.Event()
        self._pause_event.set()  # starts unpaused
        self._capture_queue = queue.Queue(maxsize=4)
        self._transcribe_queue = queue.Queue(maxsize=4)

        self._translation_buffer = deque(maxlen=1200)
        self._buffer_seq = 0  # monotonically increasing sequence number
        self._buffer_lock = threading.Lock()

        self._last_summary_text = ""

        self._transcriber = None
        self._health = HealthMonitor()

        self.on_transcript_callback = None
        self.on_translation_callback = None
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
            # Block here while paused
            while not self._pause_event.is_set() and not self._stop_event.is_set():
                self._pause_event.wait(timeout=0.5)
            if self._stop_event.is_set():
                break

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
                # segment_audio returns [] when no speech — no need for separate has_speech() call
                segments = self._vad["segment_audio"](
                    wav_path,
                    min_chunk_seconds=3.0,
                    max_chunk_seconds=15.0,
                    silence_gap_seconds=0.5,
                )

                if not segments:
                    logger.debug("No speech detected in %s, skipping", wav_path.name)
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

            text = None
            for attempt in range(3):
                try:
                    text = self._transcriber.transcribe_file(wav_path)
                    self._health.on_transcribe_success()
                    break
                except Exception as e:
                    action = self._health.on_transcribe_error(e)
                    logger.error("[TRANSCRIBE ERROR] %s (action=%s, attempt=%d/3)",
                                 e, action, attempt + 1)
                    if action == "skip":
                        break
                    time.sleep(0.5)

            if not text:
                logger.debug("Empty or failed transcript for %s", wav_path.name)
                continue

            # Filter Whisper hallucinations on silent/near-silent audio
            if SKIP_EMPTY_CHUNKS and text.strip().lower() in _HALLUCINATION_PATTERNS:
                logger.debug("Skipping hallucinated chunk: %r", text)
                continue

            self._ui.on_transcript(text)
            if self.on_transcript_callback:
                self.on_transcript_callback(text)
            line = f"[{self._ts()}] [SYS] {text}"
            self._append_line(self.out_dir / "transcript.txt", line)

            self._transcribe_queue.put((text, self._ts()))

            # Clean up WAV after successful transcription
            if CLEANUP_WAV:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError as e:
                    logger.debug("Failed to clean up %s: %s", wav_path.name, e)

    def _translate_worker(self):
        logger.info("Translation thread started (target=%s)", self.target_lang)
        MAX_TRANSLATE_RETRIES = 3

        while not self._stop_event.is_set():
            try:
                item = self._transcribe_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Support retry counter in tuple
            if len(item) == 3:
                text, timestamp, retry_count = item
            else:
                text, timestamp = item
                retry_count = 0

            # If LM Studio is down, skip translation (transcription-only mode)
            if self._health.is_lmstudio_down:
                from app.health import check_lmstudio_alive
                if check_lmstudio_alive():
                    self._health.on_llm_success()
                else:
                    continue

            try:
                if STREAM_TRANSLATION:
                    tokens = []
                    for token in translate_stream(text, target_lang=self.target_lang):
                        tokens.append(token)
                    tr = "".join(tokens).strip()
                else:
                    tr = translate(text, target_lang=self.target_lang).strip()
                self._health.on_llm_success()
            except Exception as e:
                action = self._health.on_llm_error(e)
                logger.error("[LLM ERROR] %s (action=%s, attempt=%d/%d)",
                             e, action, retry_count + 1, MAX_TRANSLATE_RETRIES)
                if action == "retry" and retry_count < MAX_TRANSLATE_RETRIES:
                    self._transcribe_queue.put((text, timestamp, retry_count + 1))
                continue

            if tr:
                self._ui.on_translation(tr, self.target_lang)
                if self.on_translation_callback:
                    self.on_translation_callback(tr)
                line_tr = f"[{self._ts()}] [SYS->{self.target_lang}] {tr}"
                self._append_line(self.out_dir / "translation.txt", line_tr)

                self._structured_output.add_entry(text, translation=tr)

                with self._buffer_lock:
                    self._buffer_seq += 1
                    self._translation_buffer.append(
                        (self._buffer_seq, time.time(), f"[{self._ts()}] {tr}")
                    )

    def _minutes_worker(self):
        logger.info("Minutes thread started (every %ds)", self.summary_every_seconds)
        last_summary_time = time.time()
        last_seen_seq = 0  # track last processed sequence number (deque-eviction safe)

        while not self._stop_event.is_set():
            elapsed = time.time() - last_summary_time
            if elapsed < self.summary_every_seconds:
                self._stop_event.wait(timeout=min(10, self.summary_every_seconds - elapsed))
                continue

            with self._buffer_lock:
                now = time.time()
                cutoff = now - MINUTES_WINDOW if MINUTES_WINDOW > 0 else 0
                new_lines = [
                    line for seq, ts, line in self._translation_buffer
                    if seq > last_seen_seq and ts >= cutoff
                ]
                current_max_seq = (
                    self._translation_buffer[-1][0]
                    if self._translation_buffer else last_seen_seq
                )

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
                last_seen_seq = current_max_seq

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

        # Preflight checks
        preflight = self._health.run_preflight()
        if not preflight.get("blackhole", False):
            logger.error("PREFLIGHT FAILED: BlackHole audio device not found. "
                         "Install BlackHole or check SYSTEM_AUDIO_IDX.")
            raise RuntimeError("BlackHole audio device not detected")
        if not preflight.get("lmstudio", False):
            logger.warning("PREFLIGHT: LM Studio not reachable — starting in transcription-only mode")
            self._health._lmstudio_was_down = True
        if not preflight.get("disk_space", False):
            logger.warning("PREFLIGHT: Low disk space — monitor carefully")

        # Clean stale WAV files from previous sessions
        if CLEANUP_WAV:
            stale = list(self.data_dir.glob("sys_*.wav"))
            if stale:
                logger.info("Cleaning %d stale WAV files from previous session", len(stale))
                for f in stale:
                    try:
                        f.unlink()
                    except OSError:
                        pass

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

        self._structured_output.flush_final()
        self._ui.stop()
        logger.info("Pipeline stopped.")

    def stop(self):
        self._stop_event.set()

    def pause(self):
        """Pause capture — queued items still drain through transcription/translation."""
        self._pause_event.clear()
        logger.info("Pipeline paused")

    def resume(self):
        """Resume capture after pause."""
        self._pause_event.set()
        logger.info("Pipeline resumed")

    def start_background(self):
        """Launch start() in a daemon thread so the caller (e.g. GUI mainloop) isn't blocked."""
        t = threading.Thread(target=self.start, name="pipeline-main", daemon=True)
        t.start()
        return t

    def reset(self):
        """Stop pipeline and clear all state so a fresh run can begin."""
        self.stop()
        # Drain queues
        for q in (self._capture_queue, self._transcribe_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        with self._buffer_lock:
            self._translation_buffer.clear()
            self._buffer_seq = 0
        self._last_summary_text = ""
        # Reset events for next run
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
