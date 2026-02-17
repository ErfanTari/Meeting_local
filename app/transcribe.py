import os
import platform
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _detect_backend():
    """Auto-detect best backend: MLX on Apple Silicon, faster-whisper elsewhere."""
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except ImportError:
            logger.info("mlx-whisper not installed, falling back to faster-whisper")
    return "faster-whisper"


class Transcriber:
    def __init__(self, model_name: str = "small", backend: str = None):
        self.backend = backend or os.getenv("WHISPER_BACKEND", None) or _detect_backend()
        self.model_name = model_name
        self._model = None

        if self.backend == "mlx":
            self._init_mlx()
        else:
            self._init_faster_whisper()

    def _init_mlx(self):
        import mlx_whisper  # noqa: F401
        self._mlx = mlx_whisper
        # mlx_whisper loads model on first transcribe call; we just store the ref
        logger.info("Transcriber: using mlx-whisper backend (model=%s)", self.model_name)

    def _init_faster_whisper(self):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        logger.info("Transcriber: using faster-whisper backend (model=%s, cpu, int8)", self.model_name)

    def transcribe_file(self, path: Path, language: str = None, task: str = "transcribe") -> str:
        """
        language: "tr" for Turkish, "en" for English, or None for auto-detect.
        task: "transcribe" for same-language, "translate" for Speech-to-English.
        """
        if self.backend == "mlx":
            return self._transcribe_mlx(path, language, task)
        return self._transcribe_faster_whisper(path, language, task)

    def _transcribe_mlx(self, path: Path, language: str, task: str) -> str:
        kwargs = {"path_or_hf_repo": f"mlx-community/whisper-{self.model_name}"}
        if language:
            kwargs["language"] = language
        if task:
            kwargs["task"] = task

        result = self._mlx.transcribe(str(path), **kwargs)
        text = result.get("text", "").strip()
        return text

    def _transcribe_faster_whisper(self, path: Path, language: str, task: str) -> str:
        segments, info = self._model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=False,
            language=language,
            task=task,
        )
        text = " ".join([s.text.strip() for s in segments]).strip()
        return text
