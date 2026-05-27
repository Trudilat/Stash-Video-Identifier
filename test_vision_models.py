#!/usr/bin/env python3
"""
Vision model comparison test.

Extracts FRAMES frames from each test video, runs them through each Ollama
vision model listed in MODELS, and writes a side-by-side report to
vision_test_output.md.

Usage:
    python test_vision_models.py
"""

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

load_dotenv()

# Allow imports from app/pipeline/
sys.path.insert(0, str(Path(__file__).parent / "app"))

from pipeline.extractor import stash_to_local, _extract_frames, cleanup, TEMP_DIR

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ── Test configuration ────────────────────────────────────────────────────────

VIDEOS = [
    # Add the Stash paths of the videos you want to test, e.g.:
    # "/data/videos/Creator/video1.mp4",
    # "/data/videos/Creator/video2.mp4",
]

MODELS = [
    "llava:7b",               # 7B
    "llava:13b",              # 13B
    "minicpm-v:latest",       # 8B
    "qwen2.5vl:7b",           # 7B
    "qwen3-vl:8b",            # 8B  — new
    "gemma3:12b",             # 12B — new
    "llava:34b",              # 34B — slow
]

FRAMES = 10
OUTPUT_FILE = Path("vision_test_output.md")

_FRAME_PROMPT = (
    "This is frame {i} of {n} from a video. "
    "Briefly describe what you see: the setting, people, activities, "
    "and any props or costumes. Be factual and concise."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_duration(local_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(local_path)],
            capture_output=True, timeout=30,
        )
        data = json.loads(r.stdout)
        dur = data.get("format", {}).get("duration")
        if dur:
            return float(dur)
    except Exception as e:
        print(f"  ffprobe failed: {e}")
    return 300.0  # safe fallback


def available_models() -> list[str]:
    """Return names of models currently available in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def unload_model(model: str) -> None:
    """Evict a model from VRAM immediately after use."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass


def describe_with_model(frames: list[Path], model: str) -> tuple[str, float]:
    """
    Send each frame to the given Ollama vision model and return
    (combined description, total seconds elapsed).
    """
    start = time.time()
    descriptions = []
    n = len(frames)

    for i, frame in enumerate(frames, start=1):
        if not frame.exists():
            continue

        print(f"      frame {i}/{n} ...", end=" ", flush=True)

        with open(frame, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": _FRAME_PROMPT.format(i=i, n=n),
                "images": [img_b64],
            }],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            text = r.json()["message"]["content"].strip()
            descriptions.append(f"**[Frame {i}/{n}]** {text}")
            print("done")
        except Exception as e:
            descriptions.append(f"**[Frame {i}/{n}]** *ERROR: {e}*")
            print(f"ERROR: {e}")

    return "\n\n".join(descriptions), time.time() - start


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Vision Model Comparison ===\n")

    # Check Ollama
    print("Checking available models in Ollama...")
    available = available_models()
    if not available:
        print("ERROR: Cannot reach Ollama. Make sure it is running.")
        sys.exit(1)

    print(f"  Available: {', '.join(available)}\n")

    models_to_test = [m for m in MODELS if m in available]
    missing        = [m for m in MODELS if m not in available]

    if missing:
        print(f"  Skipping (not installed): {', '.join(missing)}")
    if not models_to_test:
        print("ERROR: None of the configured models are installed.")
        sys.exit(1)

    print(f"  Will test: {', '.join(models_to_test)}\n")

    # Build markdown report
    lines = [
        "# Vision Model Comparison",
        "",
        f"**Models:** {', '.join(f'`{m}`' for m in models_to_test)}  ",
        f"**Frames per video:** {FRAMES}",
        "",
        "---",
        "",
    ]

    for video_path in VIDEOS:
        video_name = Path(video_path).name
        print(f"\n{'-'*60}")
        print(f"Video: {video_name}")

        local_path = stash_to_local(video_path)
        if not local_path.exists():
            print(f"  SKIP — file not found at {local_path}")
            lines += [
                f"## {video_name}",
                "",
                f"*Skipped — file not found: `{local_path}`*",
                "",
                "---",
                "",
            ]
            continue

        duration = get_duration(local_path)
        print(f"  Duration : {duration:.0f}s")
        print(f"  Extracting {FRAMES} frames...")

        work_dir = TEMP_DIR / "vision_test" / Path(video_path).stem
        work_dir.mkdir(parents=True, exist_ok=True)
        frames = _extract_frames(local_path, duration, work_dir, FRAMES)
        print(f"  Got {len(frames)} frames.\n")

        lines += [
            f"## {video_name}",
            "",
            f"*{duration:.0f}s — {len(frames)} frames*",
            "",
        ]

        for model in models_to_test:
            print(f"  [{model}]")
            description, elapsed = describe_with_model(frames, model)
            unload_model(model)
            mins, secs = divmod(int(elapsed), 60)
            elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
            per_frame   = f"{elapsed / len(frames):.1f}s/frame"
            print(f"  Done in {elapsed_str} ({per_frame})\n")

            lines += [
                f"### `{model}`",
                f"> **Time:** {elapsed_str} total &nbsp;|&nbsp; {per_frame} &nbsp;|&nbsp; {len(frames)} frames",
                "",
                description,
                "",
            ]

        cleanup(work_dir)
        lines += ["---", ""]

        # Write after each video so a crash doesn't lose progress
        OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"  (report updated)")

    print(f"\nDone. Report: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
