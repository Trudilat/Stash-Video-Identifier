import gc
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")   # "cpu" | "cuda"
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")  # "int8" | "float16" | "int8_float16"
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "8"))

# Model is loaded once and reused across all videos in a session.
# First call triggers a one-time download (~800 MB for large-v3-turbo).
# Call unload_whisper() before loading an Ollama model to free RAM.
_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        kwargs = dict(
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        if WHISPER_DEVICE == "cpu":
            kwargs["cpu_threads"] = WHISPER_THREADS
        _model = WhisperModel(WHISPER_MODEL, **kwargs)
    return _model


def transcribe(audio_path: Path, on_update=None) -> str | None:
    """
    Transcribe a WAV file and return the full transcript as a single string.
    Returns None if the audio file is missing or empty.

    on_update(text: str) is called after each segment with the accumulated
    transcript so far — use this to stream text to the UI in real time.
    """
    if audio_path is None or not audio_path.exists():
        return None
    if audio_path.stat().st_size < 1024:
        return None

    model = _get_model()

    segments, _info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,
    )

    parts = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            parts.append(text)
            if on_update:
                on_update(" ".join(parts))

    return " ".join(parts) if parts else None


def unload_whisper() -> None:
    """Release the Whisper model from RAM. Call before loading an Ollama model."""
    global _model
    if _model is not None:
        del _model
        _model = None
        gc.collect()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pipeline.extractor import extract, cleanup, stash_to_local
    from pipeline.stash_client import fetch_scenes

    folder = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Fetching a test scene from {folder!r}...")
    scenes, _ = fetch_scenes(folder=folder, min_duration=60, per_page=1)
    if not scenes:
        print("No scenes found.")
        raise SystemExit(0)

    scene = scenes[0]
    scene_id = scene["id"]
    stash_path = scene["files"][0]["path"]
    duration = scene["files"][0]["duration"]

    print(f"Scene: [{scene_id}] {stash_path} ({duration:.0f}s)")
    print("Extracting audio...")
    result = extract(scene_id, stash_path, duration)

    if result["audio"] is None:
        print("No audio track found in this video.")
        cleanup(result["work_dir"])
        raise SystemExit(0)

    print(f"Audio: {result['audio']} ({result['audio'].stat().st_size // 1024} KB)")
    print("\nTranscribing (model loads on first run)...\n")

    transcript = transcribe(result["audio"])
    cleanup(result["work_dir"])

    if transcript:
        # Print a preview — first 600 chars
        preview = transcript[:600] + ("..." if len(transcript) > 600 else "")
        print(f"Transcript preview:\n{preview}")
        print(f"\nTotal length: {len(transcript)} characters")
    else:
        print("No speech detected.")
