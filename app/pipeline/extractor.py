import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STASH_PATH_PREFIX = os.getenv("STASH_PATH_PREFIX", "/data")
LOCAL_PATH_PREFIX = Path(os.getenv("LOCAL_PATH_PREFIX", "/mnt/stash"))
TEMP_DIR         = Path(os.getenv("TEMP_DIR", "tmp"))
FRAMES_PER_VIDEO = int(os.getenv("FRAMES_PER_VIDEO", "10"))
FRAME_QUALITY    = 2  # ffmpeg -q:v: 1=best quality, 31=worst

VALID_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".m4v", ".webm", ".ts", ".mpg", ".mpeg",
}


# ---------------------------------------------------------------------------
# Path mapping
# ---------------------------------------------------------------------------

def stash_to_local(stash_path: str) -> Path:
    """
    Convert a Stash server path to a local SMB path.
    e.g. /data/videos/Creator/file.mp4 -> /mnt/stash/videos/Creator/file.mp4
    """
    rel = stash_path.removeprefix(STASH_PATH_PREFIX).lstrip("/")
    # Convert forward slashes so Path handles it correctly on Windows
    return LOCAL_PATH_PREFIX / Path(rel.replace("/", "\\"))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def is_video(stash_path: str) -> bool:
    """Return True if the file extension is a recognised video format."""
    return Path(stash_path).suffix.lower() in VALID_VIDEO_EXTENSIONS


def extract(scene_id: str, stash_path: str, duration: float,
            frames: int = None) -> dict:
    """
    Extract frames and the audio track from a video.

    frames    - number of frames to extract (defaults to FRAMES_PER_VIDEO)

    Returns:
        {
            "frames": [Path, ...],   # JPEG files
            "audio":  Path | None,   # WAV file, None if video has no audio
            "work_dir": Path,
        }

    Raises RuntimeError if the file is not a video, not found, or ffmpeg fails.
    """
    if not is_video(stash_path):
        raise RuntimeError(
            f"Skipping non-video file: {stash_path} "
            f"(extension: {Path(stash_path).suffix})"
        )

    local_path = stash_to_local(stash_path)
    if not local_path.exists():
        raise RuntimeError(f"Video file not found: {local_path}")

    work_dir = TEMP_DIR / str(scene_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    n_frames = frames if frames is not None else FRAMES_PER_VIDEO
    frame_list = _extract_frames(local_path, duration, work_dir, n_frames)
    audio = _extract_audio(local_path, work_dir)

    return {"frames": frame_list, "audio": audio, "work_dir": work_dir}


def reextract_frames(work_dir: Path, stash_path: str, duration: float,
                     frames: int) -> list[Path]:
    """
    Extract a different number of frames into an existing work directory.
    Used when the transcript has no useful content and more visual coverage
    is needed. Overwrites any previously extracted frames.
    """
    local_path = stash_to_local(stash_path)
    return _extract_frames(local_path, duration, work_dir, frames)


def cleanup(work_dir: Path) -> None:
    """Delete the temp work directory for a scene after processing is complete."""
    shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_frames(video_path: Path, duration: float, work_dir: Path,
                    n: int = None) -> list[Path]:
    """
    Extract n frames at evenly-spaced timestamps.
    Skips the first and last 5% of the video to avoid intros/outros/black frames.
    Uses input-level seeking (-ss before -i) for fast network seeks.
    """
    start = duration * 0.05
    end = duration * 0.95
    if n is None:
        n = FRAMES_PER_VIDEO
    timestamps = [start + i * (end - start) / (n - 1) for i in range(n)]

    frames = []
    for i, ts in enumerate(timestamps):
        out_path = work_dir / f"frame_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",       # fast input-level seek
            "-i", str(video_path),
            "-vframes", "1",           # exactly one frame
            "-vf", "scale=768:-2",     # resize to 768px wide for faster vision inference
            "-q:v", str(FRAME_QUALITY),
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Frame extraction failed at {ts:.1f}s\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        frames.append(out_path)

    return frames


def _extract_audio(video_path: Path, work_dir: Path) -> Path | None:
    """
    Extract audio as 16 kHz mono WAV — the format whisper.cpp expects.
    Returns None if the video has no audio track.
    """
    out_path = work_dir / "audio.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                          # drop video stream
        "-acodec", "pcm_s16le",         # 16-bit PCM
        "-ar", "16000",                 # 16 kHz
        "-ac", "1",                     # mono
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        if "no audio" in stderr.lower() or "output file is empty" in stderr.lower():
            return None
        raise RuntimeError(f"Audio extraction failed:\n{stderr}")

    # ffmpeg can succeed but produce an empty file when there's no audio track
    if not out_path.exists() or out_path.stat().st_size < 1024:
        return None

    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Verify ffmpeg is installed and on PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pipeline.stash_client import fetch_scenes

    print("=== FFmpeg check ===")
    if not check_ffmpeg():
        print("ERROR: ffmpeg not found. Install it and ensure it is on PATH.")
        raise SystemExit(1)
    print("ffmpeg OK")

    print(f"\nPath mapping: {STASH_PATH_PREFIX!r} -> {LOCAL_PATH_PREFIX}")

    # Grab the first untitled scene with a duration > 60s to test with
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    scenes, total = fetch_scenes(folder=folder, min_duration=60, per_page=1)
    if not scenes:
        print("No scenes found to test with.")
        raise SystemExit(0)

    scene = scenes[0]
    scene_id = scene["id"]
    stash_path = scene["files"][0]["path"]
    duration = scene["files"][0]["duration"]

    local = stash_to_local(stash_path)
    print(f"\nScene:      [{scene_id}] {stash_path}")
    print(f"Local path: {local}")
    print(f"Duration:   {duration:.0f}s")
    print(f"Exists:     {local.exists()}")

    if not local.exists():
        print("\nFile not found on local path. Check STASH_PATH_PREFIX / LOCAL_PATH_PREFIX in .env")
        raise SystemExit(1)

    print("\nExtracting frames and audio...")
    try:
        result = extract(scene_id, stash_path, duration)
        print(f"Frames ({len(result['frames'])}):")
        for f in result["frames"]:
            print(f"  {f}  ({f.stat().st_size // 1024} KB)")
        if result["audio"]:
            audio = result["audio"]
            print(f"Audio: {audio}  ({audio.stat().st_size // 1024} KB)")
        else:
            print("Audio: none (video has no audio track)")

        print(f"\nSuccess. Temp files in: {result['work_dir']}")
        print("Cleaning up...")
        cleanup(result["work_dir"])
        print("Done.")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
