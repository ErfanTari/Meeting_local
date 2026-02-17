from faster_whisper import WhisperModel

model_name = "small"  # later you can swap: "base", "small", "medium"
model = WhisperModel(model_name, device="cpu", compute_type="int8")

def run(path: str):
    segments, info = model.transcribe(path, beam_size=5, vad_filter=True)
    print("\n===", path, "===")
    print("language:", info.language, "prob:", info.language_probability)
    for s in segments:
        print(f"[{s.start:7.2f} -> {s.end:7.2f}] {s.text}")

run("data/system.wav")
run("data/mic.wav")
