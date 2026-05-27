import os
import requests
from dotenv import load_dotenv

load_dotenv()

STASH_URL = os.getenv("STASH_URL", "http://localhost:9999")
STASH_API_KEY = os.getenv("STASH_API_KEY", "")
GRAPHQL_URL = f"{STASH_URL}/graphql"

MIN_TAG_SCENE_COUNT = int(os.getenv("MIN_TAG_SCENE_COUNT", "15"))


def _gql(query: str, variables: dict = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if STASH_API_KEY:
        headers["ApiKey"] = STASH_API_KEY

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(GRAPHQL_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def test_connection() -> bool:
    """Returns True if Stash is reachable and the API key is valid."""
    try:
        data = _gql("{ version { version } }")
        version = data["version"]["version"]
        print(f"Connected to Stash {version}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

_ALL_TAGS_QUERY = """
query AllTags {
  allTags {
    id
    name
    scene_count
  }
}
"""


def fetch_tags(min_scene_count: int = MIN_TAG_SCENE_COUNT) -> list[dict]:
    """
    Returns tags with at least min_scene_count scenes, sorted by name.
    Each item: {"id": str, "name": str, "scene_count": int}
    """
    data = _gql(_ALL_TAGS_QUERY)
    tags = [t for t in data["allTags"] if t["scene_count"] >= min_scene_count]
    tags.sort(key=lambda t: t["name"])
    return tags


def fetch_tag_names(min_scene_count: int = MIN_TAG_SCENE_COUNT) -> list[str]:
    """Returns a sorted list of tag names meeting the scene count threshold."""
    return [t["name"] for t in fetch_tags(min_scene_count)]


def build_tag_name_to_id(min_scene_count: int = None) -> dict[str, str]:
    """Returns a mapping of tag name -> tag ID for tags meeting the threshold."""
    return {t["name"]: t["id"] for t in fetch_tags(min_scene_count or MIN_TAG_SCENE_COUNT)}


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

_FIND_SCENES_QUERY = """
query FindScenes($filter: FindFilterType, $scene_filter: SceneFilterType) {
  findScenes(filter: $filter, scene_filter: $scene_filter) {
    count
    scenes {
      id
      title
      details
      files {
        path
        duration
        size
      }
      tags {
        id
        name
      }
    }
  }
}
"""


def fetch_scenes(
    folder: str = None,
    min_duration: int = None,
    page: int = 1,
    per_page: int = 50,
) -> tuple[list[dict], int]:
    """
    Fetch scenes from Stash with optional filters.

    folder       - path substring to match (e.g. "/data/asmr/")
    min_duration - minimum duration in seconds (e.g. 60 to skip shorts)

    Returns (scenes, total_count).
    Each scene: {id, title, details, files: [{path, duration, size}], tags: [{id, name}]}

    Note: description filtering is done client-side in populate_queue because
    Stash may store unset descriptions as either NULL or empty string.
    """
    scene_filter = {}

    if folder:
        scene_filter["path"] = {"value": folder, "modifier": "INCLUDES"}

    if min_duration is not None:
        scene_filter["duration"] = {"value": min_duration, "modifier": "GREATER_THAN"}

    variables = {
        "filter": {
            "page": page,
            "per_page": per_page,
            "sort": "path",
            "direction": "ASC",
        },
        "scene_filter": scene_filter,
    }

    data = _gql(_FIND_SCENES_QUERY, variables)
    scenes = data["findScenes"]["scenes"]
    total = data["findScenes"]["count"]
    return scenes, total


def fetch_all_scenes(
    folder: str = None,
    min_duration: int = None,
    per_page: int = 100,
) -> list[dict]:
    """
    Fetches every matching scene across all pages.
    Use for building a full job queue before processing starts.
    """
    all_scenes = []
    page = 1

    while True:
        scenes, total = fetch_scenes(
            folder=folder,
            min_duration=min_duration,
            page=page,
            per_page=per_page,
        )
        all_scenes.extend(scenes)
        if len(all_scenes) >= total:
            break
        page += 1

    return all_scenes


def fetch_scene_by_id(scene_id: str) -> dict:
    """Fetch a single scene by ID."""
    query = """
    query FindScene($id: ID!) {
      findScene(id: $id) {
        id
        title
        details
        files { path duration size }
        tags { id name }
      }
    }
    """
    data = _gql(query, {"id": scene_id})
    return data["findScene"]


# ---------------------------------------------------------------------------
# Scene update
# ---------------------------------------------------------------------------

_SCENE_UPDATE_MUTATION = """
mutation SceneUpdate($input: SceneUpdateInput!) {
  sceneUpdate(input: $input) {
    id
    title
    details
    tags { id name }
  }
}
"""


def update_scene(
    scene_id: str,
    title: str = None,
    tag_ids: list[str] = None,
    description: str = None,
) -> dict:
    """
    Writes any combination of title, tags, and description to a scene in Stash.
    Only fields that are not None are included in the update — omitted fields
    are left unchanged in Stash.
    Returns the updated scene dict.
    """
    input_data = {"id": scene_id}
    if title is not None:
        input_data["title"] = title
    if tag_ids is not None:
        input_data["tag_ids"] = tag_ids
    if description is not None:
        input_data["details"] = description

    data = _gql(_SCENE_UPDATE_MUTATION, {"input": input_data})
    return data["sceneUpdate"]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== Stash connection test ===")
    if not test_connection():
        raise SystemExit(1)

    print("\n=== Tags (scene_count >= 20) ===")
    tags = fetch_tags()
    print(f"Found {len(tags)} qualifying tags")
    for t in tags[:10]:
        print(f"  [{t['id']}] {t['name']} ({t['scene_count']} scenes)")
    if len(tags) > 10:
        print(f"  ... and {len(tags) - 10} more")

    # Accept optional folder and min_duration from command line for quick testing
    # e.g. python stash_client.py /data/asmr/ 60
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    min_dur = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"\n=== Scenes needing titles (folder={folder!r}, min_duration={min_dur}s) ===")
    scenes, total = fetch_scenes(folder=folder, min_duration=min_dur, per_page=5)
    print(f"Matching scenes without a title: {total}")
    for s in scenes:
        path = s["files"][0]["path"] if s["files"] else "(no file)"
        duration = s["files"][0]["duration"] if s["files"] else 0
        print(f"  [{s['id']}] {path} ({duration:.0f}s)")
