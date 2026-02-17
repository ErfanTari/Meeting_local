"""
Voice Activity Detection using Silero VAD.
Segments continuous audio into speech chunks for more natural transcription boundaries.
"""
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Silero VAD model (loaded once, cached)
_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero VAD model loaded")
    return _vad_model, _vad_utils


def read_wav_as_tensor(wav_path: Path, sample_rate: int = 16000) -> torch.Tensor:
    """Read a WAV file and return as a float32 tensor normalized to [-1, 1]."""
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getsampwidth() == 2, "Expected 16-bit WAV"
        assert wf.getnchannels() == 1, "Expected mono WAV"
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio)


def get_speech_timestamps(
    wav_path: Path,
    sample_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    threshold: float = 0.5,
) -> List[dict]:
    """
    Detect speech segments in a WAV file.
    Returns list of dicts with 'start' and 'end' keys (sample indices).
    """
    model, utils = _load_vad()
    get_speech_ts = utils[0]  # get_speech_timestamps function

    audio = read_wav_as_tensor(wav_path, sample_rate)

    speech_timestamps = get_speech_ts(
        audio,
        model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        threshold=threshold,
    )

    return speech_timestamps


def segment_audio(
    wav_path: Path,
    sample_rate: int = 16000,
    min_chunk_seconds: float = 3.0,
    max_chunk_seconds: float = 15.0,
    silence_gap_seconds: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Segment audio into speech-based chunks.

    Returns list of (start_seconds, end_seconds) tuples representing
    speech segments suitable for transcription.

    Chunks are merged if they would be shorter than min_chunk_seconds,
    and split if longer than max_chunk_seconds.
    """
    timestamps = get_speech_timestamps(
        wav_path,
        sample_rate=sample_rate,
        min_silence_duration_ms=int(silence_gap_seconds * 1000),
    )

    if not timestamps:
        return []

    # Convert sample indices to seconds
    segments = [
        (ts["start"] / sample_rate, ts["end"] / sample_rate)
        for ts in timestamps
    ]

    # Merge short segments
    merged = []
    current_start, current_end = segments[0]

    for start, end in segments[1:]:
        duration = current_end - current_start
        gap = start - current_end

        # Merge if current chunk is too short or gap is small
        if duration < min_chunk_seconds or gap < silence_gap_seconds:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    # Split chunks that are too long
    final = []
    for start, end in merged:
        duration = end - start
        if duration <= max_chunk_seconds:
            final.append((start, end))
        else:
            # Split into roughly equal parts
            n_parts = int(np.ceil(duration / max_chunk_seconds))
            part_duration = duration / n_parts
            for i in range(n_parts):
                part_start = start + i * part_duration
                part_end = min(start + (i + 1) * part_duration, end)
                final.append((part_start, part_end))

    return final


def extract_segment_audio(wav_path: Path, start_sec: float, end_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Extract a segment of audio from a WAV file as a numpy array."""
    import wave
    with wave.open(str(wav_path), "rb") as wf:
        start_frame = int(start_sec * sample_rate)
        end_frame = int(end_sec * sample_rate)
        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    return np.frombuffer(frames, dtype=np.int16)


def has_speech(wav_path: Path, sample_rate: int = 16000, threshold: float = 0.5) -> bool:
    """Quick check: does the audio contain any speech?"""
    timestamps = get_speech_timestamps(wav_path, sample_rate=sample_rate, threshold=threshold)
    return len(timestamps) > 0
