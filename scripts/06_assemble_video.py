"""Assemble top-N detection clips into a single highlight video.

Reads detections_refined.json, sorts by quality score, takes the top-N,
and concatenates the rendered MP4 clips using FFmpeg's concat demuxer.
This is Phase 2 Step 2.1 of the simple plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ConfigError, load_config
from utils.logging_setup import configure_logging
from utils.run_archive import find_latest_run_dir

logger = logging.getLogger(__name__)


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
        logger.warning("FFmpeg timeout concatenating to %s", output_path.name)
        return False
    except Exception as exc:
        logger.warning("FFmpeg error: %s", exc)
        return False

    # Clean up the concat list file
    list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.warning("FFmpeg concat failed: %s", result.stderr.strip()[:200])
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional logfile path",
    )
    args = parser.parse_args()
    setup_signal_handlers()
    configure_logging(debug=args.debug, log_file=args.log_file)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    general = config.get("general", {})
    output_dir = Path(general.get("output_dir", "./outputs"))

    # Resolve detections input path
    detections_path = Path(args.input) if args.input else output_dir / "detections_refined.json"
    if not detections_path.exists() and not args.input:
        latest_run = find_latest_run_dir(output_dir)
        if (output_dir / "latest" / "detections_refined.json").exists():
            detections_path = output_dir / "latest" / "detections_refined.json"
        elif latest_run and (latest_run / "detections_refined.json").exists():
            detections_path = latest_run / "detections_refined.json"
    if not detections_path.exists():
        alt = Path(__file__).resolve().parent.parent / "outputs" / "detections_refined.json"
        if alt.exists():
            detections_path = alt
        else:
            logger.error(
                "detections_refined.json not found at %s. Run 02_scan_footage.py first.",
                detections_path,
            )
            sys.exit(1)

    # Resolve clips directory
    clips_dir = Path(args.clips_dir) if args.clips_dir else output_dir / "highlights"
    if not clips_dir.exists() and not args.clips_dir:
        latest_run = find_latest_run_dir(output_dir)
        if (detections_path.parent / "highlights").exists():
            clips_dir = detections_path.parent / "highlights"
        elif (output_dir / "latest" / "highlights").exists():
            clips_dir = output_dir / "latest" / "highlights"
        elif latest_run and (latest_run / "highlights").exists():
            clips_dir = latest_run / "highlights"
    if not clips_dir.exists():
        logger.error(
            "Clips directory not found: %s. Run 05_render_clips.py first to render detection clips.",
            clips_dir,
        )
        sys.exit(1)

    with open(detections_path, encoding="utf-8") as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        logger.info("No detections found to assemble.")
        sys.exit(0)

    # Sort by quality score (descending) and take top-N
    sorted_dets = sorted(detections, key=lambda d: d.get("quality_score", 0), reverse=True)
    top_dets = sorted_dets[: args.top_n]

    logger.info("Assembling top %d of %d detections", len(top_dets), len(detections))

    # Find rendered clip files matching each detection's clip_id
    clip_files: list[Path] = []
    for det in top_dets:
        raw_clip_id = str(det.get("clip_id", ""))
        clip_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_clip_id)[:64]
        raw_subject = str(det.get("subject", "unknown"))
        subject = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_subject)[:40]
        quality = det.get("quality_score", 0)

        start_time = float(det.get("start_time", 0))
        end_time = float(det.get("end_time", start_time + 5))

        # Look for rendered clip matching this detection (try timestamped first, then legacy)
        matching = list(
            clips_dir.glob(f"{clip_id}_{start_time:.1f}s-{end_time:.1f}s_{subject}_q{quality}.mp4")
        )
        if not matching:
            matching = list(clips_dir.glob(f"{clip_id}_{subject}_q{quality}.mp4"))
        if not matching:
            matching = list(clips_dir.glob(f"{clip_id}_{start_time:.1f}s-{end_time:.1f}s_*.mp4"))
        if matching:
            clip_files.append(matching[0])
            logger.info("  Using: %s (quality %s/10)", matching[0].name, quality)
        else:
            # Fallback: try any file starting with clip_id
            fallback = list(clips_dir.glob(f"{clip_id}_*.mp4"))
            if fallback:
                clip_files.append(fallback[0])
                logger.info(
                    "  Using: %s (quality %s/10, fallback match)", fallback[0].name, quality
                )
            else:
                logger.warning("  SKIP: No rendered clip found for %s (%s)", clip_id, subject)

    if not clip_files:
        logger.error("No clip files found to concatenate.")
        sys.exit(1)

    output_path = Path(args.output)
    try:
        if concat_clips(clip_files, output_path):
            logger.info("✓ Assembled %d clips into: %s", len(clip_files), output_path)
        else:
            logger.error("✗ Assembly failed.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Assembly interrupted by user (Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()
