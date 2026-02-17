from faster_whisper import WhisperModel
from pathlib import Path

class Transcriber:
    def __init__(self, model_name: str = "small"):
        # Note: Using "cpu" with "int8" is correct for your Mac Studio setup.
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe_file(self, path: Path, language: str = None, task: str = "transcribe") -> str:
        """
        language: "tr" for Turkish, "en" for English, or None for auto-detect.
        task: "transcribe" for same-language, "translate" for Speech-to-English.
        """
        segments, info = self.model.transcribe(
            str(path), 
            beam_size=5, 
            vad_filter=False,
            language=language, # Set "tr" here if you want to force Turkish
            task=task          # Use "translate" if the audio is Turkish but you want English text
        )
        
        text = " ".join([s.text.strip() for s in segments]).strip()
        return text