# Video Identifier

A local AI pipeline that analyses videos in a [Stash](https://stashapp.cc) library and writes titles, descriptions, and tags back to it. Everything runs on your own machine — no cloud services, no API keys beyond your own Stash instance.

---

## AI disclosure

This project was made almost entirely using Claude.

---

## How it works

For each untagged video the pipeline:

1. Extracts a configurable number of evenly-spaced frames and a 16 kHz mono audio track using **FFmpeg**
2. Transcribes the audio with **faster-whisper** (runs locally, CPU or CUDA)
3. Describes each frame individually using an Ollama **vision model**
4. Passes the transcript and visual descriptions to an Ollama **LLM** to generate a title, select tags from your Stash tag library, and write a short description
5. Writes the results back to Stash via its GraphQL API

Each stage can be toggled on or off independently. The pipeline tracks processed videos in a local SQLite database so interrupted runs resume where they left off.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.12 |
| [FFmpeg](https://ffmpeg.org/download.html) | Must be on `PATH` |
| [Ollama](https://ollama.com) | Runs the vision model and LLM locally |
| A Stash instance | With GraphQL API accessible |
| NVIDIA GPU *(optional)* | Speeds up Whisper transcription significantly; CPU works fine |

Ollama itself handles GPU acceleration automatically — it supports NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (Metal). The pipeline works on CPU-only machines as well, just slower.

### Python dependencies

```
pip install -r requirements.txt
```

The included `run.bat` installs dependencies automatically on first launch.

---

## Setup

### 1. Install Ollama models

You need one vision-capable model and one general LLM. For example:

```
ollama pull llava:7b
ollama pull mistral
```

Any Ollama-compatible vision model and LLM will work. Larger models generally produce better results.

### 2. Configure the environment

Copy `.env.example` to `.env` and fill in your values:

```env
# Stash connection
STASH_URL=http://localhost:9999
STASH_API_KEY=your_api_key_here

STASH_PATH_PREFIX=/data
LOCAL_PATH_PREFIX=Z:\

# Models
OLLAMA_VISION_MODEL=llava:7b
OLLAMA_LLM_MODEL=mistral

# Whisper transcription model
# Sizes: tiny | base | small | medium | large-v3-turbo
WHISPER_MODEL=large-v3-turbo
```

See `.env.example` for all available options including GPU configuration.

### Path mapping

Stash always reports file paths as the **server** sees them. If the pipeline runs on a different machine it needs to know how to translate those paths into ones valid on *this* machine.

The translation works in two steps:
1. Strip `STASH_PATH_PREFIX` from the start of the Stash path
2. Prepend `LOCAL_PATH_PREFIX`

**How to find your values:**
- Open any scene in Stash and look at the file path it shows (e.g. `/data/videos/Creator/video.mp4`). The leading portion before your actual content folders is your `STASH_PATH_PREFIX`.
- Find that same file on the machine running this pipeline and note its path. Everything before the shared folder structure is your `LOCAL_PATH_PREFIX`.

**Common setups:**

| Setup | STASH_PATH_PREFIX | LOCAL_PATH_PREFIX |
|---|---|---|
| Stash on Linux server, pipeline on Windows PC accessing files over SMB | `/data` | `Z:\` (or whichever drive letter the share is mounted as) |
| Stash in Docker, pipeline on the same host | `/data` | `/mnt/stash` (or wherever the volume is mounted on the host) |
| Stash and pipeline on the same machine, paths identical | *(leave empty)* | *(leave empty)* |

**Example — Linux server + Windows PC over SMB:**

```
Stash reports  : /data/videos/Creator/video.mp4
                  ^^^^^^ strip this
Pipeline finds : Z:\videos\Creator\video.mp4
                 ^^^ prepend this

STASH_PATH_PREFIX=/data
LOCAL_PATH_PREFIX=Z:\
```

### 3. Run

```
run.bat
```

On first launch this installs Python dependencies automatically. Ollama must already be running before you launch the app.

---

## Configuration

### In-app settings

The settings menu (option `2` from the main menu) lets you change these at runtime — they persist across sessions:

| Setting | Description |
|---|---|
| Min video duration | Skip videos shorter than this (seconds) |
| Frames per video | How many frames to extract and analyse |
| Min tag scene count | Only offer tags that appear on at least this many scenes in Stash |
| Generate title | Toggle title generation on/off |
| Generate tags | Toggle tag selection on/off |
| Generate description | Toggle description generation on/off |

### Prompt tuning

All prompts sent to the LLM are in `prompts.json` at the project root. Each prompt is stored as a plain array of strings (one per line) so they are easy to read and edit without touching any Python code.

```json
{
  "title": {
    "system": "You are a creative writer...",
    "prompt": [
      "Write a title for the adult video described below.",
      "",
      "## Transcript",
      "{transcript}",
      ...
    ]
  }
}
```

Changes take effect the next time the pipeline starts.

### Folder management

Use option `1` from the main menu to add Stash folder paths. Each folder can be configured as recursive (include subfolders) or non-recursive (immediate files only). The pipeline filters results client-side so folder matching is exact.

---

## Project structure

```
.env                 — your local configuration (not committed)
.env.example         — configuration template
prompts.json         — LLM prompts (edit to tune output)
requirements.txt     — Python dependencies
run.bat              — Windows launcher

app/
  main.py            — terminal UI and processing orchestration
  pipeline/
    stash_client.py  — Stash GraphQL API wrapper
    extractor.py     — FFmpeg frame and audio extraction
    transcriber.py   — faster-whisper transcription
    vision.py        — Ollama vision model integration
    synthesizer.py   — LLM title, tag, and description generation
    job_queue.py     — SQLite-backed job queue

.data/               — runtime state (created automatically, not committed)
  folders.json       — configured folder list
  settings.json      — persistent UI settings
  jobs.db            — processing queue and history
```

---

## Notes

- The pipeline skips videos that already have a description in Stash. Clear the description in Stash if you want a video reprocessed, then use **Retry failed jobs** or **Clear queue** and re-run.
- Duplicate files (the same video stored in multiple locations in Stash) are handled correctly — the pipeline picks the copy that belongs to the configured folder.
- Whisper is unloaded from memory before the Ollama models are loaded to avoid running out of RAM. Each Ollama model is evicted from VRAM immediately after use for the same reason.
- The `tmp/` directory is cleaned up automatically on startup and after each video is processed.

---

## License

MIT — see [LICENSE](LICENSE).
