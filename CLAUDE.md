# Meeting Local

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
meeting_local/
├── app/                          # Core library modules
│   ├── pipeline.py               # Threaded pipeline orchestrator (main engine)
│   ├── ffmpeg_capture.py         # Audio capture via ffmpeg + AVFoundation
│   ├── transcribe.py             # Whisper transcription (MLX or faster-whisper)
│   ├── lmstudio_client.py        # LM Studio API client (translate, summarize, streaming)
│   ├── vad.py                    # Voice Activity Detection (Silero VAD)
│   ├── health.py                 # Health monitoring and automatic recovery
│   ├── ui.py                     # Terminal UI (plain or rich)
│   └── output.py                 # Structured output (JSON, SRT)
├── scripts/                      # Runnable entry points
│   ├── live_loop_sys_with_minutes.py  # Main entry point (threaded pipeline)
│   ├── live_ui.py                     # Rich UI entry point
│   ├── live_loop.py                   # Legacy: system + mic capture
│   └── live_loop_sys_only.py          # Legacy: system audio only
├── tools/
│   └── doctor.py                 # Diagnostic/health check tool
├── data/live/                    # Recorded WAV chunks (generated at runtime)
├── out/                          # Output files
│   ├── transcript.txt            # Raw transcript
│   ├── translation.txt           # Translated text
│   ├── transcript.json           # Structured JSON output
│   ├── transcript.srt            # SRT subtitle format
│   ├── rolling_minutes.md        # Meeting minutes (Markdown)
│   └── rolling_minutes.txt       # Meeting minutes (plain text)
└── .venv/                        # Python 3.12 virtual environment
```

## How to Run

Always run from the project root with PYTHONPATH set:

```bash
cd /Users/erfan.tari/meeting_local

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

## Key Features

- **Pipeline parallelism**: Capture, transcription, translation, and summarization run in parallel threads
- **MLX acceleration**: Auto-detects Apple Silicon and uses MLX Whisper for ~2-3x speedup
- **VAD smart chunking**: Silero VAD detects speech boundaries — skips silence, avoids cutting words
- **Rolling summary**: Minutes use incremental summarization (previous summary + new text) instead of re-processing everything
- **Streaming LLM**: `chat_stream()` and `translate_stream()` available for real-time token output
- **Automatic recovery**: LM Studio down → transcription-only mode; ffmpeg fail → retry with backoff
- **Structured output**: JSON and SRT formats alongside plain text
- **Rich UI**: Optional split-view terminal interface with transcript + translation panels

## Key Notes

- The `app` package must be importable — always set `PYTHONPATH` to project root
- Whisper backend auto-detects: MLX on Apple Silicon, faster-whisper elsewhere
- LM Studio must be running for translation/summarization; transcription works independently
- If LM Studio goes down mid-session, the pipeline continues in transcription-only mode
- Audio device indices can change if devices are plugged/unplugged — use `tools/doctor.py` to verify
- VAD requires `torch` — if not installed, falls back to fixed chunking automatically
