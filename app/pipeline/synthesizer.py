import difflib
import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL",  "mistral")   # titles + descriptions
TAGS_MODEL  = os.getenv("OLLAMA_TAGS_MODEL",  LLM_MODEL)   # tags (falls back to LLM_MODEL)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompts() -> dict:
    """
    Load prompts from prompts.json at the project root.
    Line-arrays (["line1", "line2", ...]) are joined into single strings so
    callers can use them exactly like plain string templates.
    """
    path = Path(__file__).parents[2] / "prompts.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    def to_str(v):
        return "\n".join(v) if isinstance(v, list) else str(v)

    return {
        group: {k: to_str(v) for k, v in content.items()}
        for group, content in raw.items()
    }


_P = _load_prompts()

TRANSCRIPT_CHECK_SYSTEM_PROMPT   = _P["transcript_check"]["system"]
TRANSCRIPT_CHECK_PROMPT_TEMPLATE = _P["transcript_check"]["prompt"]
TITLE_CHECK_SYSTEM_PROMPT        = _P["title_check"]["system"]
TITLE_CHECK_PROMPT_TEMPLATE      = _P["title_check"]["prompt"]
TITLE_SYSTEM_PROMPT         = _P["title"]["system"]
TITLE_PROMPT_TEMPLATE       = _P["title"]["prompt"]
DESCRIPTION_SYSTEM_PROMPT   = _P["description"]["system"]
DESCRIPTION_PROMPT_TEMPLATE = _P["description"]["prompt"]
TAGS_SYSTEM_PROMPT          = _P["tags"]["system"]
TAGS_PROMPT_TEMPLATE        = _P["tags"]["prompt"]


# ---------------------------------------------------------------------------
# Path context helper
# ---------------------------------------------------------------------------

_GENERIC_FOLDERS = {
    "videos", "video", "data", "private", "content",
    "media", "files", "archive", "downloads", "tmp", "temp",
}


def _build_path_context(stash_path: str | None) -> str:
    if not stash_path:
        return "(no path information available)"
    p = Path(stash_path)
    lines = [f"File: {p.name}"]
    for part in reversed(p.parts[:-1]):
        clean = part.strip("/\\")
        if clean and clean.lower() not in _GENERIC_FOLDERS:
            lines.append(f"Creator/Folder: {clean}")
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _chat(system: str, user: str, temperature: float = 0.3,
          model: str = None) -> str:
    payload = {
        "model": model or LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_URL}")
    return r.json()["message"]["content"].strip()


def _unload_llm() -> None:
    """Evict both LLM models from VRAM after we are done with them."""
    for m in {LLM_MODEL, TAGS_MODEL}:
        try:
            requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": m, "keep_alive": 0},
                timeout=15,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Transcript quality check
# ---------------------------------------------------------------------------

def transcript_has_content(transcript: str | None) -> bool:
    """
    Ask the LLM whether the transcript contains meaningful spoken content.
    Returns False if empty/too short (no LLM call), or if LLM says EMPTY.
    Returns True if the transcript has useful dialogue or narration.
    """
    if not transcript or len(transcript.strip()) < 20:
        return False
    prompt = TRANSCRIPT_CHECK_PROMPT_TEMPLATE.format(
        transcript=transcript.strip()[:500]
    )
    raw = _chat(TRANSCRIPT_CHECK_SYSTEM_PROMPT, prompt, temperature=0.0)
    return "USEFUL" in raw.upper()


# ---------------------------------------------------------------------------
# Title quality check
# ---------------------------------------------------------------------------

def title_needs_replacement(title: str | None) -> bool:
    """
    Ask the LLM whether the existing title is worth keeping.
    Returns True if the title should be replaced, False if it should be kept.
    Empty/missing titles are always replaced without an LLM call.
    """
    if not title or not title.strip():
        return True
    prompt = TITLE_CHECK_PROMPT_TEMPLATE.format(title=title.strip())
    raw = _chat(TITLE_CHECK_SYSTEM_PROMPT, prompt, temperature=0.0)
    return "REPLACE" in raw.upper()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_title(
    transcript: str | None,
    visual: str | None,
    stash_path: str | None = None,
) -> str:
    """Generate a creative title for the video. Returns a plain string."""
    prompt = TITLE_PROMPT_TEMPLATE.format(
        path_context=_build_path_context(stash_path),
        transcript=transcript.strip() if transcript else "(no dialogue detected)",
        visual=visual.strip() if visual else "(no visual description available)",
    )
    # Higher temperature for the title so it's less mechanical
    raw = _chat(TITLE_SYSTEM_PROMPT, prompt, temperature=0.6)
    # Strip any stray quotes the model may add
    return raw.strip('"').strip("'").strip()


def synthesize_tags(
    transcript: str | None,
    visual: str | None,
    tag_name_to_id: dict[str, str],
) -> dict:
    """
    Select tags from tag_name_to_id that apply to this video.

    Returns:
        {"tag_ids": [str, ...], "tag_names": [str, ...]}
    """
    tag_list = "\n".join(sorted(tag_name_to_id.keys()))
    prompt = TAGS_PROMPT_TEMPLATE.format(
        transcript=transcript.strip() if transcript else "(no dialogue detected)",
        visual=visual.strip() if visual else "(no visual description available)",
        tag_list=tag_list,
    )
    raw = _chat(TAGS_SYSTEM_PROMPT, prompt, temperature=0.2, model=TAGS_MODEL)
    return _parse_tags(raw, tag_name_to_id)


def synthesize_description(
    transcript: str | None,
    visual: str | None,
    title: str,
) -> str:
    """Generate a 2-4 sentence scene description. Returns a plain string."""
    prompt = DESCRIPTION_PROMPT_TEMPLATE.format(
        title=title,
        transcript=transcript.strip() if transcript else "(no dialogue detected)",
        visual=visual.strip() if visual else "(no visual description available)",
    )
    return _chat(DESCRIPTION_SYSTEM_PROMPT, prompt, temperature=0.5)


def synthesize(
    transcript: str | None,
    visual: str | None,
    tag_name_to_id: dict[str, str],
    stash_path: str | None = None,
) -> dict:
    """
    Convenience wrapper that calls synthesize_title + synthesize_tags and
    evicts the LLM from VRAM afterwards.

    Returns:
        {"title": str, "tag_ids": [str, ...], "tag_names": [str, ...]}
    """
    title  = synthesize_title(transcript, visual, stash_path)
    tags   = synthesize_tags(transcript, visual, tag_name_to_id)
    _unload_llm()
    return {"title": title, **tags}


# ---------------------------------------------------------------------------
# Tag parsing
# ---------------------------------------------------------------------------

def _parse_tags(raw: str, tag_name_to_id: dict[str, str]) -> dict:
    text = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"No JSON found in LLM tag response:\n{raw}")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM tag JSON: {e}\nRaw:\n{raw}")

    raw_tags = data.get("tags", [])
    lower_map = {k.lower(): (k, v) for k, v in tag_name_to_id.items()}
    lower_keys = list(lower_map.keys())
    matched_names, matched_ids = [], []

    for tag in raw_tags:
        key = tag.strip().lower()
        if key in lower_map:
            name, tid = lower_map[key]
        else:
            close = difflib.get_close_matches(key, lower_keys, n=1, cutoff=0.82)
            if close:
                name, tid = lower_map[close[0]]
            else:
                continue  # skip unmatched tags silently
        matched_names.append(name)
        matched_ids.append(tid)

    return {"tag_ids": matched_ids, "tag_names": matched_names}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pipeline.extractor import extract, cleanup
    from pipeline.stash_client import fetch_scenes, build_tag_name_to_id
    from pipeline.transcriber import transcribe
    from pipeline.vision import describe_frames

    # Usage: python synthesizer.py [scene_id | folder_path]
    # e.g.   python synthesizer.py 42
    #        python synthesizer.py /data/videos/Creator/
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    print("Loading tag list from Stash...")
    tag_name_to_id = build_tag_name_to_id()
    print(f"  {len(tag_name_to_id)} tags available")

    if arg and not arg.startswith("/"):
        # Treat as a scene ID
        from pipeline.stash_client import fetch_scene_by_id
        scene = fetch_scene_by_id(arg)
    else:
        # Treat as a folder path (or None to grab any scene)
        scenes, _ = fetch_scenes(folder=arg, min_duration=60, per_page=1)
        if not scenes:
            print("No scenes found.")
            raise SystemExit(0)
        scene = scenes[0]

    scene_id = scene["id"]
    stash_path = scene["files"][0]["path"]
    duration = scene["files"][0]["duration"]
    print(f"\nScene: [{scene_id}] {stash_path} ({duration:.0f}s)")
    print(f"Path context:\n  {_build_path_context(stash_path)}\n")

    print("Extracting frames and audio...")
    result = extract(scene_id, stash_path, duration)

    print("Transcribing...")
    transcript = transcribe(result["audio"])
    preview = (transcript[:200] + "...") if transcript and len(transcript) > 200 else transcript
    print(f"  Transcript: {preview}")

    print("\nDescribing frames with vision model...")
    visual = describe_frames(result["frames"], on_update=lambda s: print(f"  {s}"))
    cleanup(result["work_dir"])

    print(f"\nSynthesizing title with {LLM_MODEL}...")
    title = synthesize_title(transcript, visual, stash_path=stash_path)
    print(f"  Title: {title}")

    print(f"\nSynthesizing tags with {LLM_MODEL}...")
    tags = synthesize_tags(transcript, visual, tag_name_to_id)
    _unload_llm()

    print("\n=== RESULT ===")
    print(f"Title : {title}")
    print(f"Tags  : {', '.join(tags['tag_names'])}")
