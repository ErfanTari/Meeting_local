# Meeting Local V2.1 — Changelog

> **V2.1** is an optimization release focused on bug fixes, performance improvements, and reliability hardening over V2.
> **Target hardware:** Mac Studio (Apple Silicon)

---

## Bug Fixes

### Minutes Buffer Index Bug (Critical)
- **Problem:** `_translation_buffer` used `deque(maxlen=1200)` with absolute index tracking. When deque evicted old entries, `all_lines[last_buffer_idx:]` silently returned wrong/empty data, causing minutes to miss transcript content in long meetings.
- **Fix:** Replaced with monotonic sequence numbers. Buffer stores `(seq_id, timestamp, line)` tuples. Minutes worker filters by `seq > last_seen_seq` — deque-eviction safe.

### VAD Double-Processing
- **Problem:** Capture worker called `has_speech()` then `segment_audio()` — both loaded the WAV and ran the Silero VAD model separately. Redundant work on every chunk.
- **Fix:** Removed `has_speech()` call. `segment_audio()` already returns `[]` for no speech.

### Translation Retry Drops Items
- **Problem:** When translation failed with "retry" action, the item was already consumed from the queue and silently lost.
- **Fix:** Failed items are re-queued as `(text, timestamp, retry_count)` with max 3 retry attempts.

---

## Performance Optimizations

### Streaming Translation
- Added `STREAM_TRANSLATION` env var. When enabled, uses `translate_stream()` for token-by-token translation, reducing time-to-first-token.
- **File:** `app/pipeline.py`

### Batched File Output
- **Problem:** `StructuredOutput.add_entry()` rewrote entire JSON + SRT files on every chunk. For long meetings, this meant hundreds of full file rewrites.
- **Fix:** Added `flush_interval=30s` — files are only written every 30 seconds. `flush_final()` ensures nothing is lost at shutdown.
- **File:** `app/output.py`

### WAV File Cleanup
- **Problem:** WAV files in `data/live/` accumulated forever (~69 MB/hour).
- **Fix:** WAVs are deleted after successful transcription. Stale files from previous sessions are cleaned on startup. Controlled by `CLEANUP_WAV` env var (default: `true`).
- **File:** `app/pipeline.py`

---

## Reliability Improvements

### Preflight Checks at Pipeline Start
- Pipeline now runs `HealthMonitor.run_preflight()` before launching threads.
- BlackHole missing = fatal error (capture would fail anyway).
- LM Studio down = starts in transcription-only mode with warning.
- Low disk space = warning.

### Transcription Retry with HealthMonitor
- Wired `_transcribe_worker` into `HealthMonitor.on_transcribe_error()` / `on_transcribe_success()`.
- Failed transcriptions retry up to 3 times with health-monitor-driven backoff.

### File Logging
- Both entry points (`live_loop_sys_with_minutes.py`, `live_ui.py`) now log to `out/meeting_local.log` alongside console output.
- Enables post-mortem debugging of pipeline issues.

### Hallucination Filter
- Added `SKIP_EMPTY_CHUNKS` (default: `true`) to filter common Whisper hallucinations on silent audio ("Thank you", "Thanks for watching", etc.).

---

## Code Quality

### Centralized Configuration
- Created `app/config.py` as single source of truth for all env var reads.
- Eliminated duplicated `LM_URL` definitions across `lmstudio_client.py` and `health.py`.

### New Environment Variables (V2.1)

| Variable | Default | Description |
|---|---|---|
| `STREAM_TRANSLATION` | `false` | Stream LLM translation responses token-by-token |
| `SKIP_EMPTY_CHUNKS` | `true` | Skip LLM calls for empty/hallucinated transcriptions |
| `MINUTES_WINDOW` | `600` | Rolling window for minutes context (seconds, 0=unlimited) |
| `CLEANUP_WAV` | `true` | Delete WAV files after successful transcription |

---

## Files Changed

| File | Change |
|---|---|
| `app/config.py` | **NEW** — Centralized env var configuration |
| `app/pipeline.py` | Bug fixes (buffer, VAD, retry), preflight, cleanup, streaming, hallucination filter |
| `app/output.py` | Batched writes with flush interval |
| `app/lmstudio_client.py` | Import from shared config |
| `app/health.py` | Import from shared config |
| `scripts/live_loop_sys_with_minutes.py` | File logging handler |
| `scripts/live_ui.py` | File logging handler |
| `CLAUDE.md` | Updated paths, added Pipeline API, new env vars, V2.1 features |
