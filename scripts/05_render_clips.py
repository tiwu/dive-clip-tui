"""Render detection windows as standalone MP4 clips via FFmpeg.

Reads detections_refined.json and extracts each highlight window as a
separate MP4 clip into a highlights folder. This is Phase 1 Step 1.3
of the simple plan.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:40]


def render_clip(
    source_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
) -> bool:
    """Extract a time window from a video file as an MP4 clip.

    Returns True on success, False on failure.
    """
    duration = end_time - start_time
    if duration <= 0:
        print(f"WARNING: Skipping {source_path.name}: non-positive duration {duration}")
        return False

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"WARNING: FFmpeg timeout rendering {output_path.name}")
        return False
    except Exception as exc:
        print(f"WARNING: FFmpeg error rendering {output_path.name}: {exc}")
        return False

    if result.returncode != 0:
        print(
            f"WARNING: FFmpeg failed for {output_path.name}: "
            f"{result.stderr.strip()[:200]}"
        )
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render AI detection windows as standalone MP4 clips"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to detections_refined.json (default: outputs/detections_refined.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output highlights folder (default: outputs/highlights)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    general = config.get("general", {})
    output_dir = Path(general.get("output_dir", "./outputs"))

    # Resolve detections input path
    detections_path = Path(args.input) if args.input else output_dir / "detections_refined.json"
    if not detections_path.exists():
        alt = Path(__file__).resolve().parent.parent / "outputs" / "detections_refined.json"
        if alt.exists():
            detections_path = alt
        else:
            print(f"ERROR: detections_refined.json not found at {detections_path}")
            print("Run 02_scan_footage.py first.")
            sys.exit(1)

    # Resolve output highlights folder
    highlights_dir = Path(args.output) if args.output else output_dir / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)

    with open(detections_path) as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        print("No detections found to render.")
        sys.exit(0)

    print(f"Rendering {len(detections)} detection clips to {highlights_dir}")

    rendered = 0
    failed = 0
    for i, det in enumerate(detections, 1):
        source_path = Path(det.get("file_path", ""))
        if not source_path.exists():
            print(f"  [{i}/{len(detections)}] SKIP: source not found: {source_path}")
            failed += 1
            continue

        start_time = float(det.get("start_time", 0))
        end_time = float(det.get("end_time", start_time + 5))
        subject = slugify(det.get("subject", "unknown"))
        quality = det.get("quality_score", 0)
        clip_id = det.get("clip_id", f"clip{i}")

        output_name = f"{clip_id}_{subject}_q{quality}.mp4"
        output_path = highlights_dir / output_name

        print(
            f"  [{i}/{len(detections)}] {source_path.name} "
            f"[{start_time:.1f}s-{end_time:.1f}s] -> {output_name}"
        )

        if render_clip(source_path, start_time, end_time, output_path):
            rendered += 1
        else:
            failed += 1

    print(
        f"\nDone. {rendered} rendered, {failed} failed."
        f"\nOutput: {highlights_dir}"
    )


if __name__ == "__main__":
    main()
