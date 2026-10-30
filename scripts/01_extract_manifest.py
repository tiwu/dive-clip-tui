"""DaVinci Resolve manifest extractor.

Runs inside DaVinci Resolve (via Workspace -> Console -> Py3).
Traverses the Media Pool and exports clips with keywords to manifest_raw.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.models import ClipManifest, ManifestOutput


def get_resolve():
    """Get the DaVinci Resolve API object."""
    try:
        import DaVinciResolveScript as bmd

        return bmd.scriptapp("Resolve")
    except ImportError:
        print("ERROR: DaVinci Resolve scripting module not available.")
        print("This script must run inside DaVinci Resolve (Workspace -> Console -> Py3).")
        return None


def resolve_path(clip: Any) -> str:
    """Get absolute file path from a Resolve clip."""
    try:
        return clip.GetClipProperty("File Path") or ""
    except Exception:
        return ""


def resolve_metadata(clip: Any, key: str) -> str:
    """Get metadata value from a Resolve clip."""
    try:
        return clip.GetMetadata(key) or ""
    except Exception:
        return ""


def resolve_property(clip: Any, prop: str) -> str:
    """Get a clip property from Resolve."""
    try:
        return clip.GetClipProperty(prop) or ""
    except Exception:
        return ""


def extract_folder(folder: Any) -> list[dict]:
    """Recursively extract clips from a Media Pool folder."""
    clips: list[dict] = []

    try:
        subfolders = folder.GetSubFolderList() or []
    except Exception:
        subfolders = []

    try:
        clip_list = folder.GetClipList() or []
    except Exception:
        clip_list = []

    for clip in clip_list:
        try:
            keywords = resolve_metadata(clip, "Keywords")
            if not keywords:
                continue

            file_path = resolve_path(clip)

            width = resolve_property(clip, "Width")
            height = resolve_property(clip, "Height")
            clip_data = {
                "clip_id": resolve_property(clip, "Clip ID")
                or hashlib.sha256((file_path + clip.GetName()).encode()).hexdigest()[:16],
                "file_name": resolve_property(clip, "File Name") or clip.GetName(),
                "file_path": file_path,
                "fps": resolve_property(clip, "FPS") or "25.0",
                "resolution": f"{width}x{height}" if width and height else "unknown",
                "tags": [k.strip() for k in keywords.split(",") if k.strip()],
                "duration": resolve_property(clip, "Duration") or "",
                "start_timecode": resolve_property(clip, "Start TC") or "",
                "end_timecode": resolve_property(clip, "End TC") or "",
            }
            clips.append(clip_data)
        except Exception as exc:
            print(f"WARNING: Failed to extract clip: {exc}")
            continue

    for subfolder in subfolders:
        try:
            clips.extend(extract_folder(subfolder))
        except Exception as exc:
            print(f"WARNING: Failed to traverse subfolder: {exc}")
            continue

    return clips


def main() -> None:
    """Main entry point."""
    resolve = get_resolve()
    if resolve is None:
        sys.exit(1)

    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        print("ERROR: No current project in DaVinci Resolve.")
        sys.exit(1)

    media_pool = project.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    if root_folder is None:
        print("ERROR: No root folder in Media Pool.")
        sys.exit(1)

    clips = extract_folder(root_folder)

    output = ManifestOutput(
        project_name=project.GetName(),
        total_clips=len(clips),
        clips=[ClipManifest(**c) for c in clips],
    )

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest_raw.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Manifest exported: {output_path} ({len(clips)} clips with keywords)")


if __name__ == "__main__":
    main()
