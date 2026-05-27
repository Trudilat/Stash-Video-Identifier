import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

# Ensure 'from pipeline.xxx import' resolves to app/pipeline/
# regardless of which directory the script is launched from.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from rich import box
except ImportError:
    print("ERROR: 'rich' is not installed. Run: pip install rich")
    sys.exit(1)

console = Console()

_DATA_DIR     = Path(".data")
_DATA_DIR.mkdir(exist_ok=True)
FOLDERS_FILE  = _DATA_DIR / "folders.json"
SETTINGS_FILE = _DATA_DIR / "settings.json"

# Defaults — overridden by settings.json if present
_DEFAULTS = {
    "min_duration":       int(os.getenv("MIN_DURATION",       "60")),
    "frames_per_video":   int(os.getenv("FRAMES_PER_VIDEO",   "10")),
    "fallback_frames":    30,
    "min_tag_count":      int(os.getenv("MIN_TAG_SCENE_COUNT", "15")),
    "generate_title":       True,
    "generate_tags":        True,
    "generate_description": True,
}

# ── Folder config ──────────────────────────────────────────────────────────

def load_folders() -> list[dict]:
    if FOLDERS_FILE.exists():
        return json.loads(FOLDERS_FILE.read_text())
    return []


def save_folders(folders: list[dict]):
    FOLDERS_FILE.write_text(json.dumps(folders, indent=2))


# ── Settings ───────────────────────────────────────────────────────────────

def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        return json.loads(SETTINGS_FILE.read_text())
    return {}


def save_settings(settings: dict):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


# ── Display ────────────────────────────────────────────────────────────────

STAGE_COLOR = {
    "Extracting":               "cyan",
    "Transcribing":             "yellow",
    "Checking transcript":      "dim yellow",
    "Extracting extra frames":  "cyan",
    "Analyzing frames":         "magenta",
    "Checking title":           "dim green",
    "Synthesizing title":       "green",
    "Synthesizing tags":        "green",
    "Synthesizing description": "green",
    "Writing to Stash":         "blue",
    "Done":                     "bold green",
}


def build_panel(state: dict) -> Panel:
    done    = state["done"]
    total   = state["total"]
    failed  = state["failed"]
    pct     = (done / total * 100) if total > 0 else 0

    bar_w   = 32
    filled  = int(bar_w * done / total) if total > 0 else 0
    bar     = f"[green]{'█' * filled}[/][dim]{'░' * (bar_w - filled)}[/]"

    fail_str = f"  [red]✗ {failed} failed[/]" if failed else ""
    progress = f"  {bar}  [bold]{done}[/] / [bold]{total}[/]  [cyan]{pct:.1f}%[/]{fail_str}"

    if state["total_video_sec"] > 0 and state["total_proc_sec"] > 0:
        ratio     = state["total_proc_sec"] / state["total_video_sec"]
        speed     = 1 / ratio  # video-minutes processed per real minute
        remaining = state["remaining_video_sec"] * ratio
        eta       = str(timedelta(seconds=int(remaining))) if remaining > 0 else "—"
        speed_str = f"x{speed:.1f}"
        speed_line = f"  Speed [yellow]{speed_str}[/]  ·  ETA [yellow]{eta}[/]"
    else:
        speed_line = "  Speed [dim]—[/]  ·  ETA [dim]calculating...[/]"

    stage = state.get("stage", "")
    color = STAGE_COLOR.get(stage, "white")

    if state.get("current_path"):
        fname    = state["current_path"].split("/")[-1]
        dur      = int(state.get("current_duration", 0))
        dur_str  = f"{dur // 60}m {dur % 60}s"
        now_line = f"  [bold]Now:[/] [white]{fname}[/]  [dim]({dur_str})[/]"
        stage_line = f"  Stage: [bold {color}]{stage}[/]"
    else:
        now_line   = "  [dim]Idle[/]"
        stage_line = ""

    transcript = state.get("transcript", "")
    if len(transcript) > 350:
        transcript = "[dim]...[/]" + transcript[-350:]
    transcript_block = (
        f"  [italic]{transcript}[/]" if transcript
        else "  [dim italic](no dialogue detected)[/]"
    )

    if state.get("last_title"):
        tags_str = "  [dim]·[/]  ".join(f"[cyan]{t}[/]" for t in state["last_tags"])
        result_block = (
            f"  [bold]Title[/]  [green]{state['last_title']}[/]\n"
            f"  [bold]Tags[/]   {tags_str or '[dim]none[/]'}"
        )
    else:
        result_block = "  [dim italic](no results yet)[/]"

    sep = "[dim]─[/]" * 60

    content = "\n".join([
        progress,
        speed_line,
        f"  {sep}",
        now_line,
        stage_line,
        f"  {sep}",
        "  [bold underline]Transcript[/]",
        transcript_block,
        f"  {sep}",
        "  [bold underline]Last Result[/]",
        result_block,
    ])

    return Panel(content, title="[bold magenta]Video Identifier[/]",
                 border_style="bright_blue", padding=(0, 1))


# ── Diagnostics ────────────────────────────────────────────────────────────

def run_diagnostics():
    console.print(Rule("[bold]Diagnostics[/]", style="bright_blue"))

    # 1. Stash
    console.print("\n[cyan]1. Stash API[/]")
    try:
        from pipeline.stash_client import test_connection
        ok = test_connection()
        console.print("   [green]PASS[/]" if ok else "   [red]FAIL[/]")
    except Exception as e:
        console.print(f"   [red]FAIL[/] — {e}")

    # 2. FFmpeg
    console.print("\n[cyan]2. FFmpeg[/]")
    try:
        from pipeline.extractor import check_ffmpeg
        ok = check_ffmpeg()
        console.print("   [green]PASS[/]" if ok else "   [red]FAIL — ffmpeg not found on PATH[/]")
    except Exception as e:
        console.print(f"   [red]FAIL[/] — {e}")

    # 3. Whisper
    console.print("\n[cyan]3. Whisper model[/]")
    try:
        from pipeline.transcriber import _get_model
        _get_model()
        console.print("   [green]PASS[/] — model loaded")
    except Exception as e:
        console.print(f"   [red]FAIL[/] — {e}")

    # 4. Ollama / Vision model
    console.print("\n[cyan]4. Ollama + vision model[/]")
    try:
        from pipeline.vision import check_ollama
        ok = check_ollama()
        console.print("   [green]PASS[/]" if ok else "   [red]FAIL[/]")
    except Exception as e:
        console.print(f"   [red]FAIL[/] — {e}")

    # 5. Ollama / LLM
    console.print("\n[cyan]5. Ollama + LLM[/]")
    try:
        import requests, os
        url   = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_LLM_MODEL", "")
        r = requests.post(f"{url}/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one word: OK"}],
            "stream": False,
            "options": {"num_predict": 5},
        }, timeout=60)
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
        console.print(f"   [green]PASS[/] — model replied: [italic]{reply}[/]")
    except Exception as e:
        console.print(f"   [red]FAIL[/] — {e}")

    console.print()
    Prompt.ask("\nPress Enter to return to menu")


# ── Folder management ──────────────────────────────────────────────────────

def manage_folders():
    while True:
        folders = load_folders()
        console.clear()
        console.print(Rule("[bold]Folder Management[/]", style="bright_blue"))

        if folders:
            t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            t.add_column("#",         style="dim",   width=4)
            t.add_column("Stash Path",               min_width=40)
            t.add_column("Recursive", justify="center", width=10)
            for i, f in enumerate(folders, 1):
                rec = "[green]Yes[/]" if f["recursive"] else "[yellow]No[/]"
                t.add_row(str(i), f["path"], rec)
            console.print(t)
        else:
            console.print("  [dim]No folders configured.[/]\n")

        console.print("  [bold]add[/]       — Add a folder")
        console.print("  [bold]remove N[/]  — Remove folder N")
        console.print("  [bold]toggle N[/]  — Toggle recursive for folder N")
        console.print("  [bold]back[/]      — Return to main menu\n")

        cmd = Prompt.ask(">").strip().lower()

        if cmd == "back":
            break
        elif cmd == "add":
            path = Prompt.ask("  Stash path (e.g. /data/videos/Creator/)").strip()
            if not path:
                continue
            if not path.endswith("/"):
                path += "/"
            rec = Confirm.ask("  Recursive?", default=True)
            folders.append({"path": path, "recursive": rec})
            save_folders(folders)
            console.print(f"  [green]Added:[/] {path} (recursive={rec})")
            time.sleep(0.8)
        elif cmd.startswith("remove "):
            try:
                n = int(cmd.split()[1]) - 1
                removed = folders.pop(n)
                save_folders(folders)
                console.print(f"  [red]Removed:[/] {removed['path']}")
                time.sleep(0.8)
            except (IndexError, ValueError):
                console.print("  [red]Invalid number.[/]")
                time.sleep(0.8)
        elif cmd.startswith("toggle "):
            try:
                n = int(cmd.split()[1]) - 1
                folders[n]["recursive"] = not folders[n]["recursive"]
                save_folders(folders)
                state = "Yes" if folders[n]["recursive"] else "No"
                console.print(f"  [cyan]Recursive toggled to {state}[/]")
                time.sleep(0.8)
            except (IndexError, ValueError):
                console.print("  [red]Invalid number.[/]")
                time.sleep(0.8)


# ── Queue population ───────────────────────────────────────────────────────

def populate_queue(folders: list[dict], settings: dict) -> int:
    from pipeline.stash_client import fetch_all_scenes
    from pipeline.extractor import is_video
    from pipeline.job_queue import add_jobs, matches_folder

    total_added = 0
    for folder_cfg in folders:
        path = folder_cfg["path"]
        recursive = folder_cfg["recursive"]
        console.print(f"  Fetching scenes from [cyan]{path}[/] (recursive={recursive})...")

        scenes = fetch_all_scenes(
            folder=path,
            min_duration=settings["min_duration"],
        )

        # Client-side filters.
        # Scenes can have multiple files (duplicates in different locations).
        # Find the file that actually belongs to the configured folder so that
        # folder matching and later extraction use the right copy.
        filtered = []
        for s in scenes:
            if not s["files"]:
                continue
            if (s.get("details") or "").strip():
                continue  # already has a description

            matching_file = next(
                (f for f in s["files"]
                 if is_video(f["path"]) and matches_folder(f["path"], path, recursive)),
                None,
            )
            if matching_file is None:
                continue

            # Put the matching file first so add_jobs and the pipeline use it
            if matching_file != s["files"][0]:
                s = {**s, "files": [matching_file] + [f for f in s["files"] if f is not matching_file]}
            filtered.append(s)

        added = add_jobs(filtered)
        console.print(f"    {len(filtered)} scenes matched, [green]{added}[/] new jobs queued.")
        total_added += added

    return total_added


# ── Processing loop ────────────────────────────────────────────────────────

def process_one(job: dict, tag_map: dict, settings: dict, state: dict, live: Live):
    from pipeline.extractor import extract, cleanup
    from pipeline.transcriber import transcribe, unload_whisper
    from pipeline.vision import describe_frames
    from pipeline.synthesizer import (
        synthesize_title, synthesize_tags, synthesize_description,
        title_needs_replacement, _unload_llm,
    )
    from pipeline.stash_client import update_scene
    from pipeline.job_queue import mark_done, mark_failed

    scene_id  = job["scene_id"]
    stash_path = job["stash_path"]
    duration  = job["duration"]
    start     = time.time()

    state.update({"current_path": stash_path, "current_duration": duration, "transcript": ""})

    def update(stage):
        state["stage"] = stage
        live.update(build_panel(state))

    result = None
    try:
        update("Extracting")
        result = extract(scene_id, stash_path, duration,
                         frames=settings["frames_per_video"])

        update("Transcribing")
        def on_seg(text):
            state["transcript"] = text
            live.update(build_panel(state))

        transcript = transcribe(result["audio"], on_update=on_seg)
        unload_whisper()

        # If transcript has no useful content and fallback is configured,
        # re-extract more frames before running the vision model.
        from pipeline.synthesizer import transcript_has_content
        from pipeline.extractor import reextract_frames

        frames_to_describe = result["frames"]
        fallback = settings.get("fallback_frames", 30)
        if fallback > 0:
            update("Checking transcript")
            if not transcript_has_content(transcript):
                _unload_llm()  # free LLM VRAM before loading vision model
                update("Extracting extra frames")
                frames_to_describe = reextract_frames(
                    result["work_dir"], stash_path, duration, fallback
                )
            else:
                _unload_llm()

        def on_frame(stage_text):
            state["stage"] = stage_text
            live.update(build_panel(state))

        visual = describe_frames(frames_to_describe, on_update=on_frame)
        cleanup(result["work_dir"])
        result = None

        existing_title = job.get("existing_title", "") or ""
        title      = None   # None = don't update in Stash
        tag_result = {"tag_ids": None, "tag_names": []}
        description = None

        if settings.get("generate_title", True):
            update("Checking title")
            if title_needs_replacement(existing_title):
                update("Synthesizing title")
                title = synthesize_title(transcript, visual, stash_path=stash_path)
            else:
                title = existing_title

        if settings.get("generate_tags", True):
            update("Synthesizing tags")
            tag_result = synthesize_tags(transcript, visual, tag_map)

        if settings.get("generate_description", True):
            update("Synthesizing description")
            description = synthesize_description(
                transcript, visual, title or existing_title
            )

        _unload_llm()

        update("Writing to Stash")
        update_scene(
            scene_id,
            title=title,
            tag_ids=tag_result["tag_ids"],
            description=description,
        )

        proc_sec = time.time() - start
        display_title = title or existing_title
        mark_done(scene_id, display_title, tag_result["tag_names"], proc_sec)

        state["done"]            += 1
        state["total_proc_sec"]  += proc_sec
        state["total_video_sec"] += duration
        state["remaining_video_sec"] = max(0, state["remaining_video_sec"] - duration)
        state["last_title"]      = display_title
        state["last_tags"]       = tag_result["tag_names"]
        state["stage"]           = "Done"
        live.update(build_panel(state))

    except Exception as e:
        if result and result.get("work_dir"):
            cleanup(result["work_dir"])
        mark_failed(scene_id, str(e))
        state["failed"] += 1
        state["remaining_video_sec"] = max(0, state["remaining_video_sec"] - duration)
        console.print(f"\n  [red]Failed[/] [{scene_id}]: {e}")


def run_processing(settings: dict):
    from pipeline.job_queue import (
        init_db, reset_in_progress, get_next_job, get_counts
    )
    from pipeline.stash_client import build_tag_name_to_id

    folders = load_folders()
    if not folders:
        console.print("[red]No folders configured. Add folders first.[/]")
        time.sleep(1.5)
        return

    console.print("\nFetching tag list from Stash...")
    tag_map = build_tag_name_to_id(min_scene_count=settings["min_tag_count"])
    console.print(f"  [green]{len(tag_map)}[/] tags loaded.\n")

    console.print("Populating job queue...")
    init_db()
    reset_in_progress()
    populate_queue(folders, settings)
    console.print()

    counts = get_counts()
    if counts["pending"] == 0 and counts["done"] == 0:
        console.print("[yellow]No jobs to process. Queue is empty.[/]")
        time.sleep(1.5)
        return

    state = {
        "done":               counts["done"] or 0,
        "total":              counts["total"] or 0,
        "failed":             counts["failed"] or 0,
        "total_proc_sec":     counts["total_proc_sec"] or 0.0,
        "total_video_sec":    counts["total_video_sec"] or 0.0,
        "remaining_video_sec": counts["remaining_video_sec"] or 0.0,
        "stage":              "",
        "current_path":       "",
        "current_duration":   0,
        "transcript":         "",
        "last_title":         "",
        "last_tags":          [],
    }

    console.print(
        f"Queue: [bold]{state['total']}[/] total  "
        f"([green]{state['done']} done[/] / [yellow]{counts['pending']} pending[/] / "
        f"[red]{state['failed']} failed[/])\n"
    )
    console.print("[dim]Press Ctrl+C to stop gracefully.[/]\n")
    time.sleep(1)

    with Live(build_panel(state), console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                job = get_next_job()
                if job is None:
                    break
                process_one(job, tag_map, settings, state, live)
        except KeyboardInterrupt:
            live.update(build_panel(state))
            console.print("\n[yellow]Stopped by user.[/]")

    console.print(
        f"\n[bold green]Finished.[/] "
        f"{state['done']} done, {state['failed']} failed."
    )
    Prompt.ask("\nPress Enter to return to menu")


# ── Main menu ──────────────────────────────────────────────────────────────

def manage_settings(settings: dict):
    """Interactive submenu for editing persistent settings."""
    # (key, label, type, extra)
    # type "int"  → extra = (unit, min, max)
    # type "bool" → extra = None  (selection toggles immediately)
    items = [
        ("min_duration",        "Min video duration",   "int",  ("s",       1, 86400)),
        ("frames_per_video",    "Frames per video",     "int",  (" frames", 1, 100)),
        ("fallback_frames",     "Fallback frame count", "int",  (" frames (0 = off)", 0, 100)),
        ("min_tag_count",       "Min tag scene count",  "int",  (" scenes", 1, 10000)),
        ("generate_title",      "Generate title",       "bool", None),
        ("generate_tags",       "Generate tags",        "bool", None),
        ("generate_description","Generate description", "bool", None),
    ]

    while True:
        console.clear()
        console.print(Rule("[bold]Settings[/]", style="bright_blue"))
        console.print()

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("num",  style="bold", width=4)
        t.add_column("name", style="cyan")
        t.add_column("val",  justify="left")
        for i, (key, label, kind, extra) in enumerate(items, 1):
            if kind == "bool":
                val_str = "[green]On[/]" if settings[key] else "[dim]Off[/]"
            else:
                unit = extra[0]
                val_str = f"{settings[key]}{unit}"
            t.add_row(f"[{i}]", label, val_str)
        console.print(t)

        console.print("  Enter a number to change a value, or [bold]back[/] to return.\n")
        cmd = Prompt.ask(">").strip().lower()

        if cmd == "back":
            break

        try:
            idx = int(cmd) - 1
            key, label, kind, extra = items[idx]
        except (ValueError, IndexError):
            console.print("  [red]Invalid choice.[/]")
            time.sleep(0.8)
            continue

        if kind == "bool":
            settings[key] = not settings[key]
            save_settings(settings)
            state = "[green]On[/]" if settings[key] else "[dim]Off[/]"
            console.print(f"  {label}: {state}")
            time.sleep(0.6)
        else:
            unit, lo, hi = extra
            val = Prompt.ask(f"  {label} (current: {settings[key]}{unit})")
            try:
                n = int(val)
                if not (lo <= n <= hi):
                    raise ValueError
                settings[key] = n
                save_settings(settings)
                console.print(f"  [green]Set to {n}{unit}[/]")
                time.sleep(0.6)
            except ValueError:
                console.print(f"  [red]Must be a whole number between {lo} and {hi}.[/]")
                time.sleep(0.8)


def _cleanup_tmp():
    """Delete any leftover temp files from a previous interrupted run."""
    import shutil
    tmp = Path(os.getenv("TEMP_DIR", "tmp"))
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    from pipeline.job_queue import init_db, reset_in_progress, get_counts

    _cleanup_tmp()
    init_db()
    reset_in_progress()

    settings = {**_DEFAULTS, **load_settings()}
    save_settings(settings)  # persist any newly added defaults

    while True:
        console.clear()
        console.print(Rule("[bold magenta]Video Identifier[/]", style="bright_blue"))
        console.print()

        try:
            counts = get_counts()
            console.print(
                f"  Queue: [bold]{counts['total']}[/] total  "
                f"([green]{counts['done']} done[/] / "
                f"[yellow]{counts['pending']} pending[/] / "
                f"[red]{counts['failed']} failed[/])"
            )
        except Exception:
            console.print("  Queue: [dim]empty[/]")

        def _onoff(key):
            return "[green]On[/]" if settings[key] else "[dim]Off[/]"

        folders = load_folders()
        fallback_str = str(settings['fallback_frames']) if settings['fallback_frames'] > 0 else "Off"
        console.print(f"  Folders: [cyan]{len(folders)} configured[/]  ·  "
                      f"Min duration: [cyan]{settings['min_duration']}s[/]  ·  "
                      f"Frames: [cyan]{settings['frames_per_video']}[/]  ·  "
                      f"Fallback: [cyan]{fallback_str}[/]  ·  "
                      f"Min tag count: [cyan]{settings['min_tag_count']}[/]")
        console.print(f"  Title: {_onoff('generate_title')}  ·  "
                      f"Tags: {_onoff('generate_tags')}  ·  "
                      f"Description: {_onoff('generate_description')}")
        console.print()
        console.print("  [bold][1][/] Manage folders")
        console.print("  [bold][2][/] Settings")
        console.print("  [bold][3][/] Run diagnostics")
        console.print("  [bold][4][/] Start processing")
        console.print("  [bold][5][/] Retry failed jobs")
        console.print("  [bold][6][/] Clear queue")
        console.print("  [bold][7][/] Exit")
        console.print()

        choice = Prompt.ask(">").strip()

        if choice == "1":
            manage_folders()
        elif choice == "2":
            manage_settings(settings)
        elif choice == "3":
            console.clear()
            run_diagnostics()
        elif choice == "4":
            console.clear()
            run_processing(settings)
        elif choice == "5":
            from pipeline.job_queue import reset_failed
            n = reset_failed()
            if n:
                console.print(f"  [green]{n} failed job(s) reset to pending.[/]")
            else:
                console.print("  [dim]No failed jobs to retry.[/]")
            time.sleep(1)
        elif choice == "6":
            if Confirm.ask("  Clear all jobs from the queue?", default=False):
                from pipeline.job_queue import clear_queue
                clear_queue()
                console.print("  [green]Queue cleared.[/]")
                time.sleep(1)
        elif choice == "7":
            console.print("\n[dim]Goodbye.[/]")
            break


if __name__ == "__main__":
    main()
