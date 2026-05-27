#!/usr/bin/env python3
"""
LLM model comparison test — title & tag generation.

Phase 1  Extract visual descriptions (qwen3-vl:8b) and transcripts (Whisper)
         from each test video once. Results are cached to llm_test_cache.json
         so Phase 2 can be re-run without re-extracting.

Phase 2  Run every installed LLM through the identical title + tags prompts
         and record output and timing. Report written to llm_test_output.md.

Usage:
    python test_llm_models.py               # full run
    python test_llm_models.py --no-extract  # skip Phase 1, use existing cache
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "app"))

from pipeline.extractor import (
    stash_to_local, _extract_frames, _extract_audio, cleanup, TEMP_DIR
)
from pipeline.transcriber import transcribe, unload_whisper
from pipeline.vision import describe_frames, unload_model as unload_vision

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ── Configuration ─────────────────────────────────────────────────────────────

VIDEOS = [
    # Add the Stash paths of the videos you want to test, e.g.:
    # "/data/videos/Creator/video1.mp4",
    # "/data/videos/Creator/video2.mp4",
]

MODELS = [
    "mistral:latest",
    "qwen2.5:7b",
    "llama3.1:8b",
    "phi4:latest",
    "gemma3:12b",
    # Add any other models you have installed, e.g.:
    # "your-custom-model:tag",
]

# Optional short display names for long model names
MODEL_ALIASES: dict[str, str] = {
    # "your-custom-model:tag": "Short Name",
}

FRAMES       = 10
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")
CACHE_FILE   = Path("llm_test_cache.json")
OUTPUT_FILE  = Path("llm_test_output.md")

# ── Fallback tag list (used when Stash is unreachable) ────────────────────────

SAMPLE_TAGS = sorted([
    "Anal", "Anal Play", "Anal Toy", "Ass",
    "BDSM", "Bathroom", "Bedroom", "Blindfold", "Bondage",
    "Boot Fetish", "Boots", "Butt Plug",
    "Choker", "Close Up", "Collar", "Corset",
    "Dildo", "Dominant", "Domination",
    "Face Sitting", "Feet", "Femdom", "Fetish", "Fingering",
    "Fishnet", "Foot Fetish", "Foot Worship", "Footjob",
    "Gag", "Girl on Girl", "Gloves",
    "Handcuffs", "Harness", "High Heels", "Humiliation",
    "Latex", "Leather", "Leash", "Lesbian", "Lingerie",
    "Masturbation", "Moaning",
    "Oral", "Orgasm", "Outdoor",
    "POV", "Pierced", "Power Play",
    "Rimming", "Role Play", "Rough",
    "Sex Toy", "Shoe Fetish", "Shower", "Solo", "Spanking",
    "Stockings", "Strap-on", "Submissive",
    "Tattoos", "Tease", "Thigh Highs",
    "Vibrator", "Voyeur",
    "Whip", "Worship",
])

# ── Prompt loading ─────────────────────────────────────────────────────────────

def _load_prompts() -> dict:
    path = Path(__file__).parent / "prompts.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def to_str(v):
        return "\n".join(v) if isinstance(v, list) else str(v)

    return {
        group: {k: to_str(v) for k, v in content.items()}
        for group, content in raw.items()
    }


_P           = _load_prompts()
TITLE_SYSTEM = _P["title"]["system"]
TITLE_PROMPT = _P["title"]["prompt"]
TAGS_SYSTEM  = _P["tags"]["system"]
TAGS_PROMPT  = _P["tags"]["prompt"]

# ── Path context ──────────────────────────────────────────────────────────────

_GENERIC = {
    "videos", "video", "data", "private", "content",
    "media", "files", "archive", "downloads", "tmp", "temp",
}


def _path_context(stash_path: str) -> str:
    p = Path(stash_path)
    lines = [f"File: {p.name}"]
    for part in reversed(p.parts[:-1]):
        clean = part.strip("/\\")
        if clean and clean.lower() not in _GENERIC:
            lines.append(f"Creator/Folder: {clean}")
            break
    return "\n".join(lines)


# ── Ollama helpers ────────────────────────────────────────────────────────────

def available_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _unload(model: str) -> None:
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=15,
        )
    except Exception:
        pass


def _chat(model: str, system: str, user: str, temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


# ── Tag list ──────────────────────────────────────────────────────────────────

def get_tag_list() -> str:
    try:
        from pipeline.stash_client import build_tag_name_to_id
        tags = build_tag_name_to_id()
        if tags:
            print(f"  Fetched {len(tags)} tags from Stash.")
            return "\n".join(sorted(tags.keys()))
    except Exception as e:
        print(f"  Stash unreachable ({e}), using sample tag list.")
    return "\n".join(SAMPLE_TAGS)


# ── Phase 1: extraction ───────────────────────────────────────────────────────

def get_duration(local_path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(local_path)],
            capture_output=True, timeout=30,
        )
        return float(json.loads(r.stdout).get("format", {}).get("duration", 300))
    except Exception:
        return 300.0


def extract_video_data(stash_path: str) -> dict:
    """Extract frames → visual description + audio → transcript for one video."""
    local_path = stash_to_local(stash_path)
    work_dir   = TEMP_DIR / "llm_test" / Path(stash_path).stem
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(local_path)
    print(f"    Duration: {duration:.0f}s")

    # Visual
    print(f"    Extracting {FRAMES} frames...", flush=True)
    frames = _extract_frames(local_path, duration, work_dir, FRAMES)
    print(f"    Describing with {VISION_MODEL}...", flush=True)
    t0     = time.time()
    visual = describe_frames(frames)
    unload_vision(VISION_MODEL)
    print(f"    Visual done in {time.time() - t0:.0f}s")

    # Transcript
    print(f"    Extracting audio...", flush=True)
    audio      = _extract_audio(local_path, work_dir)
    transcript = ""
    if audio:
        print(f"    Transcribing...", flush=True)
        t0         = time.time()
        transcript = transcribe(audio) or ""
        print(f"    Transcript done in {time.time() - t0:.0f}s")
    else:
        print(f"    No audio track.")
    unload_whisper()

    cleanup(work_dir)

    return {
        "name":         Path(stash_path).name,
        "stash_path":   stash_path,
        "path_context": _path_context(stash_path),
        "visual":       visual,
        "transcript":   transcript,
    }


# ── Phase 2: LLM test ─────────────────────────────────────────────────────────

def test_model(model: str, video_data: dict, tag_list: str) -> dict:
    """Run title + tags generation for one model/video pair."""
    transcript = video_data.get("transcript") or "(no dialogue detected)"
    visual     = video_data.get("visual")     or "(no visual description available)"
    path_ctx   = video_data.get("path_context", "")

    # Title
    title_user = TITLE_PROMPT.format(
        path_context=path_ctx,
        transcript=transcript,
        visual=visual,
    )
    t0 = time.time()
    try:
        title = _chat(model, TITLE_SYSTEM, title_user, temperature=0.6)
        title = title.strip('"').strip("'").strip()
    except Exception as e:
        title = f"*ERROR: {e}*"
    title_time = time.time() - t0

    # Tags
    tags_user = TAGS_PROMPT.format(
        transcript=transcript,
        visual=visual,
        tag_list=tag_list,
    )
    t0 = time.time()
    try:
        raw  = _chat(model, TAGS_SYSTEM, tags_user, temperature=0.2)
        text = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        tags = json.loads(m.group()).get("tags", []) if m else []
    except Exception as e:
        tags = [f"*ERROR: {e}*"]
    tags_time = time.time() - t0

    return {
        "title":      title,
        "tags":       tags,
        "title_time": title_time,
        "tags_time":  tags_time,
        "total_time": title_time + tags_time,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    skip_extract = "--no-extract" in sys.argv

    print("=== LLM Model Comparison — Title & Tags ===\n")

    available = available_models()
    if not available:
        print("ERROR: Cannot reach Ollama.")
        sys.exit(1)

    models_to_test = [m for m in MODELS if m in available]
    missing        = [m for m in MODELS if m not in available]
    if missing:
        print(f"Skipping (not installed): {', '.join(missing)}")
    if not models_to_test:
        print("ERROR: None of the configured models are installed.")
        sys.exit(1)

    aliases = [MODEL_ALIASES.get(m, m) for m in models_to_test]
    print(f"Models to test ({len(models_to_test)}): {', '.join(aliases)}\n")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if skip_extract and CACHE_FILE.exists():
        print(f"Loading cached data from {CACHE_FILE}...")
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        print(f"  {len(cache)} video(s) cached.\n")
    else:
        print("Phase 1: Extracting video data (frames + transcripts)\n")
        cache = {}
        for vpath in VIDEOS:
            name = Path(vpath).name
            print(f"  {name}")
            local = stash_to_local(vpath)
            if not local.exists():
                print(f"    SKIP — not found at {local}\n")
                continue
            cache[vpath] = extract_video_data(vpath)
            print()

        CACHE_FILE.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Cache saved -> {CACHE_FILE}\n")

    if not cache:
        print("ERROR: No videos could be processed.")
        sys.exit(1)

    # ── Tag list ──────────────────────────────────────────────────────────────
    print("Getting tag list...")
    tag_list = get_tag_list()
    print()

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print("Phase 2: LLM comparison\n")

    lines = [
        "# LLM Model Comparison — Title & Tags",
        "",
        f"**Models tested:** {', '.join(f'`{a}`' for a in aliases)}  ",
        f"**Vision model (extraction):** `{VISION_MODEL}`  ",
        f"**Frames per video:** {FRAMES}  ",
        f"**Tag source:** {'Stash' if len(tag_list.splitlines()) > len(SAMPLE_TAGS) else 'sample list'}",
        "",
        "---",
        "",
    ]

    for vpath, video_data in cache.items():
        name = video_data["name"]
        print(f"{'='*60}")
        print(f"Video: {name}\n")

        # Show condensed input in the report
        transcript_preview = (
            video_data["transcript"][:400] + "..."
            if len(video_data.get("transcript", "")) > 400
            else (video_data.get("transcript") or "*(none)*")
        )
        visual_preview = video_data["visual"][:600] + "..." if video_data.get("visual") else "*(none)*"

        lines += [
            f"## {name}",
            "",
            "<details><summary>Shared input (click to expand)</summary>",
            "",
            "**Path context:**",
            "```",
            video_data["path_context"],
            "```",
            "",
            f"**Transcript:** {transcript_preview}",
            "",
            f"**Visual (truncated):** {visual_preview}",
            "",
            "</details>",
            "",
        ]

        for model in models_to_test:
            display = MODEL_ALIASES.get(model, model)
            print(f"  [{display}] ...", end=" ", flush=True)

            result = test_model(model, video_data, tag_list)
            _unload(model)

            mins, secs = divmod(int(result["total_time"]), 60)
            elapsed    = f"{mins}m {secs}s" if mins else f"{secs}s"
            print(f"done in {elapsed}")

            tag_str = ", ".join(result["tags"]) if result["tags"] else "*(none)*"
            lines += [
                f"### `{display}`",
                f"> **Time:** {elapsed} &nbsp;|&nbsp; title {result['title_time']:.1f}s "
                f"&nbsp;|&nbsp; tags {result['tags_time']:.1f}s",
                "",
                f"**Title:** {result['title']}",
                "",
                f"**Tags ({len(result['tags'])}):** {tag_str}",
                "",
            ]

        lines += ["---", ""]
        OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  (report updated)\n")

    print(f"Done. Report: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
