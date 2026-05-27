import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")


def unload_model(model: str = None) -> None:
    """Immediately evict a model from VRAM by setting keep_alive=0."""
    model = model or VISION_MODEL
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass  # best-effort, don't crash the pipeline

_FRAME_PROMPT = (
    "This is frame {i} of {n} from a video. "
    "Briefly describe what you see: the setting, people, activities, and any props or costumes. "
    "Be factual and concise."
)


def describe_frames(frames: list[Path], on_update=None) -> str | None:
    """
    Describe each frame individually then return a combined visual summary.
    Returns None if no frames are provided.

    on_update(stage: str) is called before each frame so the caller can show
    live progress (e.g. "Analyzing frames 3/10") without printing to stdout.
    """
    if not frames:
        return None

    descriptions = []
    n = len(frames)

    for i, frame in enumerate(frames, start=1):
        if not frame.exists():
            continue

        if on_update:
            on_update(f"Analyzing frames {i}/{n}")

        with open(frame, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": _FRAME_PROMPT.format(i=i, n=n),
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            text = response.json()["message"]["content"].strip()
            descriptions.append(f"[Frame {i}/{n}] {text}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_URL}. Is Ollama running?"
            )
        except Exception as e:
            raise RuntimeError(f"Vision model failed on frame {i}: {e}")

    if not descriptions:
        return None

    unload_model()
    return "\n\n".join(descriptions)


def check_ollama(model: str = None) -> bool:
    """Check Ollama is reachable and the requested model is available."""
    model = model or VISION_MODEL
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        available = [m["name"] for m in response.json().get("models", [])]
        if model not in available:
            print(f"Model '{model}' not found. Available: {available}")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to Ollama at {OLLAMA_URL}")
        return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pipeline.extractor import extract, cleanup
    from pipeline.stash_client import fetch_scenes

    print("=== Ollama vision check ===")
    if not check_ollama():
        raise SystemExit(1)
    print(f"Model '{VISION_MODEL}' ready.\n")

    folder = sys.argv[1] if len(sys.argv) > 1 else None
    scenes, _ = fetch_scenes(folder=folder, min_duration=60, per_page=1)
    if not scenes:
        print("No scenes found.")
        raise SystemExit(0)

    scene = scenes[0]
    scene_id = scene["id"]
    stash_path = scene["files"][0]["path"]
    duration = scene["files"][0]["duration"]

    print(f"Scene: [{scene_id}] {stash_path} ({duration:.0f}s)")
    print("Extracting frames...")
    result = extract(scene_id, stash_path, duration)
    print(f"Sending {len(result['frames'])} frames to {VISION_MODEL}...\n")

    description = describe_frames(result["frames"])
    cleanup(result["work_dir"])

    print("=== VISUAL DESCRIPTION ===")
    print(description)
    print("=== END ===")
