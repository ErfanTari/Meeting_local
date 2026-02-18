# Meeting Local — Future Improvement Ideas

> Ideas collected during V2.1 optimization. Pick and prioritize based on real usage pain points.

---

## Performance

### Install mlx-whisper
- Doctor flagged it's missing. Mac Studio has Apple Silicon — MLX gives ~2-3x faster transcription.
- `pip install mlx-whisper` in the venv, then it auto-detects via `WHISPER_BACKEND`.

### Parallel VAD segment processing
- Currently VAD segments from one chunk are queued one-by-one into the transcription queue.
- Could start transcribing segment 0 while still extracting segment 1 from the same chunk.
- Minor win but adds up over long meetings.

### Larger/faster Whisper model
- With MLX, `medium` or `large-v3` may be fast enough on Mac Studio while giving much better accuracy.
- Benchmark with `WHISPER_MODEL=medium` and compare quality vs latency.

---

## Features

### Speaker diarization
- Detect who is speaking so minutes show "Speaker 1 said X, Speaker 2 said Y".
- Options: `pyannote-audio`, `whisperX`, or NeMo MSDD.
- Would require combining system audio + mic audio with speaker embeddings.
- High value for multi-person meetings.

### Language auto-detection with translation skip
- Whisper returns detected language. If source language already matches `TARGET_LANG`, skip the LLM translation call entirely.
- Saves ~2-4s per chunk when people speak in the target language.
- Easy to implement: check `info.language` from faster-whisper or `result["language"]` from MLX.

### Web dashboard
- Single-file HTML served via `python -m http.server` or a minimal Flask/FastAPI app.
- Polls `out/transcript.json` and renders live transcript + translation.
- Useful for non-terminal users following along on a second screen.
- Stretch: WebSocket for real-time push instead of polling.

### Mic + system audio merge
- `live_loop.py` already captures both mic and system audio separately.
- Merging them with speaker labels ("Remote" vs "Local") would give a complete meeting record.
- Could run two capture threads in parallel and tag transcripts by source.

### Keyboard shortcuts / hotkeys
- Global hotkey to pause/resume capture (e.g., when stepping away).
- Pipeline already has `pause()` / `resume()` methods — just needs a listener.
- Options: `pynput` for global hotkeys, or integrate into the Rich UI.

### Export formats
- PDF export of rolling minutes for sharing after meetings.
- Email/Slack integration to send minutes automatically when pipeline stops.

---

## Reliability

### Graceful WAV cleanup on failed transcription
- Currently if transcription fails after all retries, the WAV file stays in `data/live/` forever.
- Add a cleanup sweep on pipeline shutdown for any remaining WAVs.

### Session separation
- Timestamp output files per session (e.g., `transcript_20260218_1430.txt`).
- Multiple meetings in the same day currently overwrite each other.
- Could create a session subfolder in `out/` per run.

### Auto-restart on crash
- Wrap pipeline in a supervisor that restarts on unexpected crash.
- Or use `launchd` / systemd-style service definition for always-on use.

### Health check endpoint
- Expose a simple HTTP endpoint (e.g., `localhost:8765/health`) that returns pipeline status.
- Useful for monitoring tools or the web dashboard to show if pipeline is alive.

---

## Code Quality

### Move away from Anaconda Python
- Doctor flagged the current Python is Anaconda-packaged.
- A clean `brew install python` or `pyenv install 3.13` avoids potential native-lib conflicts with torch/mlx/onnxruntime.

### Type hints and tests
- Add type hints to pipeline.py public API methods.
- Unit tests for VAD segmentation logic, hallucination filter, buffer sequence tracking.
- Integration test: dry-run with a sample WAV through the full pipeline.

### Package as a proper Python project
- Add `pyproject.toml` with dependencies so `pip install -e .` works.
- Eliminates the need for `PYTHONPATH=.` on every run.
- Makes it easier to share or deploy.

---

## Priority Suggestion

| Improvement | Impact | Effort |
|---|---|---|
| Install mlx-whisper | High (2-3x speed) | 5 min |
| Language auto-skip | Medium (saves LLM calls) | 1 hour |
| Session separation | Medium (data hygiene) | 1 hour |
| Speaker diarization | Very high (meeting quality) | 2-3 days |
| Web dashboard | Medium (usability) | 1-2 days |
| pyproject.toml packaging | Medium (DX) | 1 hour |
