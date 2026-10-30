"""Render detection windows as standalone MP4 clips via FFmpeg.

Reads detections_refined.json and extracts each highlight window as a
separate MP4 clip into a highlights folder. This is Phase 1 Step 1.3
of the simple plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.checkpoint import PipelineState
from utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)


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
        logger.warning("Skipping %s: non-positive duration %s", source_path.name, duration)
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Running FFmpeg command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timeout rendering %s", output_path.name)
        return False
    except Exception as exc:
        logger.warning("FFmpeg error rendering %s: %s", output_path.name, exc)
        return False

    if result.returncode != 0:
        logger.warning("FFmpeg failed for %s: %s", output_path.name, result.stderr.strip()[:200])
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render AI detection windows as standalone MP4 clips"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
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
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Pipeline state file to heartbeat (default: not written)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append this run's log lines to the given file",
    )
    args = parser.parse_args()

    configure_logging(debug=args.debug, log_file=args.log_file)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
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
            logger.error("detections_refined.json not found at %s", detections_path)
            logger.error("Run 02_scan_footage.py first.")
            sys.exit(1)

    # Resolve output highlights folder
    highlights_dir = Path(args.output) if args.output else output_dir / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)

    with open(detections_path) as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        logger.info("No detections found to render.")
        sys.exit(0)

    logger.info("Rendering %d detection clips to %s", len(detections), highlights_dir)

    state: PipelineState | None = None
    if args.state_path is not None:
        state = PipelineState(args.state_path)
        state.mark_step_started("render")

    rendered = 0
    failed = 0
    for i, det in enumerate(detections, 1):
        source_path = Path(det.get("file_path", ""))
        if not source_path.exists():
            logger.warning("  [%d/%d] SKIP: source not found: %s", i, len(detections), source_path)
            failed += 1
            continue

        start_time = float(det.get("start_time", 0))
        end_time = float(det.get("end_time", start_time + 5))
        subject = slugify(det.get("subject", "unknown"))
        quality = det.get("quality_score", 0)
        clip_id = det.get("clip_id", f"clip{i}")

        output_name = f"{clip_id}_{subject}_q{quality}.mp4"
        output_path = highlights_dir / output_name

        if state is not None:
            state.set_activity(f"render: {i}/{len(detections)} {output_name}")

        logger.info(
            "  [%d/%d] %s [%.1fs-%.1fs] -> %s",
            i,
            len(detections),
            source_path.name,
            start_time,
            end_time,
            output_name,
        )

        clip_start = time.perf_counter()
        if render_clip(source_path, start_time, end_time, output_path):
            rendered += 1
            logger.debug(
                "  [%d/%d] rendered in %.1fs: %s",
                i,
                len(detections),
                time.perf_counter() - clip_start,
                output_name,
            )
        else:
            failed += 1

    if state is not None:
        state.mark_step_done("render")

    logger.info("Done. %d rendered, %d failed.", rendered, failed)
    logger.info("Output: %s", highlights_dir)


if __name__ == "__main__":
    main()
