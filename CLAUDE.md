# Meeting Local V2.1

Real-time meeting transcription and translation tool for macOS. Captures system audio via BlackHole, transcribes with Whisper (MLX or faster-whisper), translates via local LLM (LM Studio), and generates rolling meeting minutes.

## Architecture

The pipeline uses a **threaded architecture** for parallel processing:

```
[Capture Thread] → capture_queue → [Transcribe Thread] → transcribe_queue → [Translate Thread]
                                                                                    ↓
                                                                          translation_buffer
                                                                                    ↓
                                                                          [Minutes Thread]
```

Each stage runs independently, connected by thread-safe queues. This reduces latency from ~12-18s (sequential) to ~2-4s per chunk.

## Project Structure

```
meeting_local_V2/
├── app/                          # Core library modules
│   ├── config.py                 # Shared configuration (all env var reads)
│   ├── pipeline.py               # Threaded pipeline orchestrator (main engine)
│   ├── ffmpeg_capture.py         # Audio capture via ffmpeg + AVFoundation
│   ├── transcribe.py             # Whisper transcription (MLX or faster-whisper)
│   ├── lmstudio_client.py        # LM Studio API client (translate, summarize, streaming)
│   ├── vad.py                    # Voice Activity Detection (Silero VAD)
│   ├── health.py                 # Health monitoring and automatic recovery
│   ├── ui.py                     # Terminal UI (plain or rich)
│   └── output.py                 # Structured output (JSON, SRT) with batched writes
├── scripts/                      # Runnable entry points
│   ├── live_loop_sys_with_minutes.py  # Main entry point (threaded pipeline)
│   ├── live_ui.py                     # Rich UI entry point
│   ├── live_loop.py                   # Legacy: system + mic capture
│   └── live_loop_sys_only.py          # Legacy: system audio only
├── tools/
│   └── doctor.py                 # Diagnostic/health check tool
├── data/live/                    # Recorded WAV chunks (auto-cleaned after transcription)
├── out/                          # Output files
│   ├── transcript.txt            # Raw transcript
│   ├── translation.txt           # Translated text
│   ├── transcript.json           # Structured JSON output
│   ├── transcript.srt            # SRT subtitle format
│   ├── rolling_minutes.md        # Meeting minutes (Markdown)
│   ├── rolling_minutes.txt       # Meeting minutes (plain text)
│   └── meeting_local.log         # Session log
└── .venv/                        # Python 3.13 virtual environment
```

## How to Run

Always run from the project root with PYTHONPATH set:

```bash
cd /Users/erfan.tari/meeting_local_V2

# Main pipeline (plain text output)
PYTHONPATH=. .venv/bin/python scripts/live_loop_sys_with_minutes.py

# Rich terminal UI
RICH_UI=true PYTHONPATH=. .venv/bin/python scripts/live_ui.py

# Dry run (test pipeline without live audio)
PYTHONPATH=. .venv/bin/python scripts/live_loop_sys_with_minutes.py --dry-run

# Doctor check
PYTHONPATH=. .venv/bin/python tools/doctor.py
```

## Prerequisites

- **ffmpeg**: `brew install ffmpeg`
- **BlackHole 2ch**: Virtual audio device for system audio capture
- **LM Studio**: Running at `http://localhost:1234` with a model loaded
- **Python venv**: `.venv/` with dependencies

### Python Dependencies
- `faster-whisper`, `ctranslate2` — Whisper transcription (default backend)
- `mlx-whisper` — MLX Whisper backend (optional, recommended on Apple Silicon)
- `requests` — HTTP client for LM Studio
- `torch`, `numpy` — Required for VAD (optional)
- `rich` — Rich terminal UI (optional)

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SYSTEM_AUDIO_IDX` | `2` | AVFoundation audio device index for system audio (BlackHole) |
| `MIC_AUDIO_IDX` | `0` | AVFoundation audio device index for microphone |
| `CHUNK_SECONDS` | `10` | Audio chunk duration in seconds |
| `TARGET_LANG` | `English` | Translation target language |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `WHISPER_BACKEND` | auto-detect | `mlx` or `faster-whisper` (auto-detects Apple Silicon) |
| `VAD_ENABLED` | `true` | Enable Silero VAD for smart chunking |
| `RICH_UI` | `false` | Enable rich terminal UI with split view |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `LMSTUDIO_MODEL_FAST` | `google/gemma-3-4b` | Model for translation |
| `LMSTUDIO_MODEL_SMART` | `google/gemma-3-4b` | Model for meeting minutes |
| `SUMMARY_EVERY_SECONDS` | `300` | Interval for rolling minutes generation |
| `STREAM_TRANSLATION` | `false` | Stream LLM translation responses token-by-token |
| `SKIP_EMPTY_CHUNKS` | `true` | Skip LLM calls for empty/hallucinated transcriptions |
| `MINUTES_WINDOW` | `600` | Rolling window for minutes context (seconds, 0=unlimited) |
| `CLEANUP_WAV` | `true` | Delete WAV files after successful transcription |

## Pipeline API

The `Pipeline` class in `app/pipeline.py` provides these methods:

| Method | Description |
|---|---|
| `start()` | Run preflight checks, start all threads, block until stop |
| `stop()` | Signal all threads to stop |
| `pause()` | Pause audio capture (queued items continue processing) |
| `resume()` | Resume audio capture after pause |
| `reset()` | Stop pipeline, drain queues, clear buffers for a fresh run |
| `start_background()` | Launch `start()` in a daemon thread (for GUI integration) |

## Key Features

- **Pipeline parallelism**: Capture, transcription, translation, and summarization run in parallel threads
- **MLX acceleration**: Auto-detects Apple Silicon and uses MLX Whisper for ~2-3x speedup
- **VAD smart chunking**: Silero VAD detects speech boundaries — skips silence, avoids cutting words
- **Rolling summary**: Minutes use incremental summarization (previous summary + new text) with time-windowed context
- **Streaming LLM**: Optional streaming translation via `STREAM_TRANSLATION=true`
- **Automatic recovery**: LM Studio down → transcription-only mode; ffmpeg fail → retry with backoff; transcription errors → retry with health monitor
- **Preflight checks**: Pipeline validates BlackHole, LM Studio, and disk space before starting
- **Hallucination filter**: Skips common Whisper hallucinations on silent audio
- **Structured output**: JSON and SRT formats with batched writes (every 30s) to reduce I/O
- **WAV cleanup**: Auto-deletes audio chunks after transcription; cleans stale files on startup
- **Rich UI**: Optional split-view terminal interface with transcript + translation panels
- **File logging**: Session log at `out/meeting_local.log` for post-mortem debugging

## Key Notes

- The `app` package must be importable — always set `PYTHONPATH` to project root
- Whisper backend auto-detects: MLX on Apple Silicon, faster-whisper elsewhere
- LM Studio must be running for translation/summarization; transcription works independently
- If LM Studio goes down mid-session, the pipeline continues in transcription-only mode
- Audio device indices can change if devices are plugged/unplugged — use `tools/doctor.py` to verify
- VAD requires `torch` — if not installed, falls back to fixed chunking automatically
- All env vars are centralized in `app/config.py` — single source of truth
