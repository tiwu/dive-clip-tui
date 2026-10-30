"""Folder ingestion: walk a directory, enumerate videos, extract metadata, emit JSON manifest.

This is Phase 1 Step 1.1 of the simple plan. Points at a folder of videos
and produces a plain JSON manifest with duration, FPS, resolution, and codec.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg"}


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:40]


def probe_video(video_path: Path) -> dict:
    """Extract duration, FPS, resolution, and codec from a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {
                "file_name": video_path.name,
                "file_path": str(video_path),
                "error": result.stderr.strip() or "ffprobe failed",
            }

        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        duration = float(fmt.get("duration", 0)) if fmt.get("duration") else 0.0
        fps_str = stream.get("r_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            fps = round(float(num) / float(den), 2) if float(den) > 0 else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0

        return {
            "file_name": video_path.name,
            "file_path": str(video_path),
            "duration": round(duration, 2),
            "fps": fps,
            "resolution": f"{stream.get('width', '?')}x{stream.get('height', '?')}",
            "codec": stream.get("codec_name", "unknown"),
        }

    except subprocess.TimeoutExpired:
        return {
            "file_name": video_path.name,
            "file_path": str(video_path),
            "error": "ffprobe timeout",
        }
    except Exception as exc:
        return {
            "file_name": video_path.name,
            "file_path": str(video_path),
            "error": str(exc),
        }


def ingest_folder(folder: Path) -> list[dict]:
    """Walk a folder and collect metadata for all video files."""
    return [
        probe_video(video_path)
        for video_path in sorted(folder.rglob("*"))
        if video_path.is_file()
        and not video_path.name.startswith("._")
        and video_path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a folder of videos and emit a JSON manifest"
    )
    parser.add_argument("--input", required=True, help="Path to footage folder")
    parser.add_argument(
        "--output", default="outputs/manifest_raw.json", help="Output manifest JSON path"
    )
    args = parser.parse_args()

    folder = Path(args.input)
    if not folder.exists():
        print(f"ERROR: Folder does not exist: {folder}")
        sys.exit(1)

    clips = ingest_folder(folder)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project_name": folder.name,
        "total_clips": len(clips),
        "clips": clips,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest written: {output_path} ({len(clips)} clips)")


if __name__ == "__main__":
    main()
