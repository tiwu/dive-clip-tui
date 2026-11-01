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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ConfigError, load_config
from utils.ffmpeg_streamer import concat_clips
from utils.logging_setup import configure_logging
from utils.run_archive import resolve_and_validate_assembly_artifacts
from utils.signals import setup_signal_handlers

logger = logging.getLogger(__name__)


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
        default=None,
        help="Number of top-quality clips to include (default: 10, or config)",
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

    # Resolve detections input path and clips directory
    try:
        detections_path, clips_dir, _ = resolve_and_validate_assembly_artifacts(
            output_dir,
            input_detections=args.input,
            clips_dir=args.clips_dir,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    with open(detections_path, encoding="utf-8") as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        logger.info("No detections found to assemble.")
        sys.exit(0)

    # Sort by quality score (descending) and take top-N (CLI flag > config > default 10)
    aftermovie_cfg = (
        config.get("aftermovie", {}) if isinstance(config.get("aftermovie"), dict) else {}
    )
    top_n = args.top_n if args.top_n is not None else int(aftermovie_cfg.get("top_n_clips", 10))
    sorted_dets = sorted(detections, key=lambda d: d.get("quality_score", 0), reverse=True)
    top_dets = sorted_dets[:top_n]

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
