"""Assemble top-N detection clips into a single highlight video.

Reads detections_refined.json, sorts by quality score, takes the top-N,
and concatenates the rendered MP4 clips using FFmpeg's concat demuxer.
This is Phase 2 Step 2.1 of the simple plan.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ConfigError, load_config


def setup_signal_handlers() -> None:
    """Ensure SIGINT and SIGTERM raise KeyboardInterrupt for clean shutdown."""

    def _handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except ValueError, AttributeError:
        pass


def concat_clips(clip_paths: list[Path], output_path: Path) -> bool:
    """Concatenate MP4 clips using FFmpeg's concat demuxer.

    All clips must have identical codec/resolution (they do, since they're
    all rendered by 05_render_clips.py with the same settings).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg concat demuxer requires absolute or relative paths in the list file.
    list_path = output_path.parent / "concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for clip in clip_paths:
            path_str = str(clip.resolve())
            if "\n" in path_str or "\r" in path_str:
                raise ValueError(f"Path contains newline character: {path_str}")
            # FFmpeg concat demuxer escaping for single-quoted string: escape \ then '
            escaped = path_str.replace("\\", "\\\\").replace("'", "\\'")
            f.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"WARNING: FFmpeg timeout concatenating to {output_path.name}")
        return False
    except Exception as exc:
        print(f"WARNING: FFmpeg error: {exc}")
        return False

    # Clean up the concat list file
    list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"WARNING: FFmpeg concat failed: {result.stderr.strip()[:200]}")
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble top-N detection clips into a single highlight video"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to detections_refined.json (default: outputs/detections_refined.json)",
    )
    parser.add_argument(
        "--clips-dir",
        default=None,
        help="Folder with rendered clips (default: outputs/highlights)",
    )
    parser.add_argument(
        "--output",
        default="outputs/aftermovie.mp4",
        help="Output video file (default: outputs/aftermovie.mp4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top-quality clips to include (default: 10)",
    )
    args = parser.parse_args()
    setup_signal_handlers()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

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

    # Resolve clips directory
    clips_dir = Path(args.clips_dir) if args.clips_dir else output_dir / "highlights"
    if not clips_dir.exists():
        print(f"ERROR: Clips directory not found: {clips_dir}")
        print("Run 05_render_clips.py first to render detection clips.")
        sys.exit(1)

    with open(detections_path) as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        print("No detections found to assemble.")
        sys.exit(0)

    # Sort by quality score (descending) and take top-N
    sorted_dets = sorted(detections, key=lambda d: d.get("quality_score", 0), reverse=True)
    top_dets = sorted_dets[: args.top_n]

    print(f"Assembling top {len(top_dets)} of {len(detections)} detections")

    # Find rendered clip files matching each detection's clip_id
    clip_files: list[Path] = []
    for det in top_dets:
        raw_clip_id = str(det.get("clip_id", ""))
        clip_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_clip_id)[:64]
        raw_subject = str(det.get("subject", "unknown"))
        subject = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_subject)[:40]
        quality = det.get("quality_score", 0)

        # Look for rendered clip matching this detection
        matching = list(clips_dir.glob(f"{clip_id}_{subject}_q{quality}.mp4"))
        if matching:
            clip_files.append(matching[0])
            print(f"  Using: {matching[0].name} (quality {quality}/10)")
        else:
            # Fallback: try any file starting with clip_id
            fallback = list(clips_dir.glob(f"{clip_id}_*.mp4"))
            if fallback:
                clip_files.append(fallback[0])
                print(f"  Using: {fallback[0].name} (quality {quality}/10, fallback match)")
            else:
                print(f"  SKIP: No rendered clip found for {clip_id} ({subject})")

    if not clip_files:
        print("No clip files found to concatenate.")
        sys.exit(1)

    output_path = Path(args.output)
    try:
        if concat_clips(clip_files, output_path):
            print(f"\n✓ Assembled {len(clip_files)} clips into: {output_path}")
        else:
            print("\n✗ Assembly failed.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nAssembly interrupted by user (Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()
