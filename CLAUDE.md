# Meeting Local

Real-time meeting transcription and translation tool for macOS. Captures system audio via BlackHole, transcribes with Whisper, translates via local LLM (LM Studio).

## Project Structure

```
Meeting_local/
├── app/                    # Core library modules
│   ├── ffmpeg_capture.py   # Audio capture via ffmpeg + AVFoundation
│   ├── transcribe.py       # Whisper transcription (faster-whisper, cpu/int8)
│   └── lmstudio_client.py  # LM Studio OpenAI-compatible API client
├── scripts/                # Runnable entry points
│   ├── live_loop.py                  # Basic: system + mic capture, translate
│   ├── live_loop_sys_only.py         # System audio only, translate
│   └── live_loop_sys_with_minutes.py # System audio + periodic meeting minutes (main script)
├── tools/
│   └── doctor.py           # Diagnostic/health check tool
├── data/live/              # Recorded WAV chunks (generated at runtime)
├── out/                    # Output files (transcript.txt, translation.txt, rolling_minutes.md)
└── .venv/                  # Python 3.12 virtual environment
```

## How to Run

Always run from the project root with PYTHONPATH set:

```bash
cd /Users/erfan.tari/Meeting_local
PYTHONPATH=/Users/erfan.tari/Meeting_local .venv/bin/python scripts/live_loop_sys_with_minutes.py
```

Or run the doctor check:
```bash
PYTHONPATH=/Users/erfan.tari/Meeting_local .venv/bin/python tools/doctor.py
```

## Prerequisites

- **ffmpeg**: `brew install ffmpeg`
- **BlackHole 2ch**: Virtual audio device for system audio capture (device index 2)
- **LM Studio**: Running at `http://localhost:1234` with a model loaded (default: `google/gemma-3-4b`)
- **Python venv**: `.venv/` with `faster-whisper`, `requests`, `ctranslate2`

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SYSTEM_AUDIO_IDX` | `2` | AVFoundation audio device index for system audio (BlackHole) |
| `MIC_AUDIO_IDX` | `0` | AVFoundation audio device index for microphone |
| `CHUNK_SECONDS` | `10` | Audio chunk duration in seconds |
| `OVERLAP_SECONDS` | `2` | Overlap between chunks (live_loop_sys_with_minutes) |
| `TARGET_LANG` | `English` | Translation target language |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `LMSTUDIO_MODEL_FAST` | `google/gemma-3-4b` | Model for translation |
| `LMSTUDIO_MODEL_SMART` | `google/gemma-3-4b` | Model for meeting minutes |
| `SUMMARY_EVERY_SECONDS` | `300` | Interval for rolling minutes generation |

## Key Notes

- The `app` package must be importable — always set `PYTHONPATH` to project root or run with `-m`
- Whisper runs on CPU with int8 quantization (optimized for Mac Studio)
- LM Studio must be running with a model loaded for translation/summarization to work; transcription works independently
- Audio device indices can change if devices are plugged/unplugged — use `tools/doctor.py` to verify
