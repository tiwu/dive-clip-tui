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
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.checkpoint import PipelineState
from utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def setup_signal_handlers() -> None:
    """Ensure SIGINT and SIGTERM raise KeyboardInterrupt for clean shutdown."""

    def _handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except ValueError, AttributeError:
        pass


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:40]


def render_clip(
    source_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    log_to_console: bool = True,
) -> bool:
    """Extract a time window from a video file as an MP4 clip.

    Returns True on success, False on failure.
    """
    extra = {"no_console": True} if not log_to_console else {}
    duration = end_time - start_time
    if duration <= 0:
        logger.warning(
            "Skipping %s: non-positive duration %s",
            source_path.name,
            duration,
            extra=extra,
        )
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
            logger.debug("Running FFmpeg command: %s", " ".join(cmd), extra=extra)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timeout rendering %s", output_path.name, extra=extra)
        return False
    except KeyboardInterrupt:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        logger.warning("FFmpeg error rendering %s: %s", output_path.name, exc, extra=extra)
        return False

    if result.returncode != 0:
        logger.warning(
            "FFmpeg failed for %s: %s",
            output_path.name,
            result.stderr.strip()[:200],
            extra=extra,
        )
        return False

    return True


def render_detections(
    detections: list[dict | Any],
    highlights_dir: Path,
    state: PipelineState | None = None,
    log_to_console: bool = True,
) -> tuple[int, int]:
    """Render detection windows into standalone MP4 clips.

    Skips clips whose destination files already exist with non-zero size.
    Returns (rendered_count, failed_count).
    """
    highlights_dir = Path(highlights_dir)
    highlights_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    failed = 0
    extra = {"no_console": True} if not log_to_console else {}
    for i, item in enumerate(detections, 1):
        det = item.model_dump() if hasattr(item, "model_dump") else item
        source_path = Path(det.get("file_path", ""))
        if not source_path.exists():
            logger.warning(
                "  [%d/%d] SKIP: source not found: %s",
                i,
                len(detections),
                source_path,
                extra=extra,
            )
            failed += 1
            continue

        start_time = float(det.get("start_time", 0))
        end_time = float(det.get("end_time", start_time + 5))
        subject = slugify(det.get("subject", "unknown"))
        quality = det.get("quality_score", 0)
        clip_id = slugify(str(det.get("clip_id", f"clip{i}")))

        output_name = f"{clip_id}_{subject}_q{quality}.mp4"
        output_path = highlights_dir / output_name

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.debug(
                "  [%d/%d] SKIP: already rendered: %s",
                i,
                len(detections),
                output_name,
                extra=extra,
            )
            continue

        if state is not None:
            state.set_activity(f"render: {i}/{len(detections)} {output_name}")

        msg = (
            f"  [{i}/{len(detections)}] {source_path.name} "
            f"[{start_time:.1f}s-{end_time:.1f}s] -> {output_name}"
        )
        if not log_to_console:
            msg = (
                f"  [{i}/{len(detections)}] (background) {source_path.name} "
                f"[{start_time:.1f}s-{end_time:.1f}s] -> {output_name}"
            )

        logger.info(msg, extra=extra)

        clip_start = time.perf_counter()
        if log_to_console:
            ok = render_clip(source_path, start_time, end_time, output_path)
        else:
            ok = render_clip(
                source_path,
                start_time,
                end_time,
                output_path,
                log_to_console=False,
            )

        if ok:
            rendered += 1
            logger.debug(
                "  [%d/%d] rendered in %.1fs: %s",
                i,
                len(detections),
                time.perf_counter() - clip_start,
                output_name,
                extra=extra,
            )
        else:
            failed += 1

    return rendered, failed


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
    setup_signal_handlers()

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

    try:
        rendered, failed = render_detections(detections, highlights_dir, state=state)
    except KeyboardInterrupt:
        logger.warning("\nRender interrupted by user (Ctrl+C).")
        if state is not None:
            state.set_activity("render: interrupted by user")
        sys.exit(130)

    concurrent_ready = len(detections) - rendered - failed

    if state is not None:
        state.mark_step_done("render")
        total_highlight_duration = sum(
            max(0.0, float(d.get("end_time", 0)) - float(d.get("start_time", 0)))
            for d in detections
        )
        render_duration = float(state.get("render_duration_seconds", 0.0) or 0.0)
        state.set(
            "render_stats",
            {
                "clips_total": len(detections),
                "clips_rendered": rendered + concurrent_ready,
                "clips_skipped": concurrent_ready,
                "clips_failed": failed,
                "total_duration": round(total_highlight_duration, 2),
                "elapsed_seconds": render_duration,
            },
        )
        state.save()

    if concurrent_ready > 0:
        logger.info(
            "Done. %d clips ready (%d newly rendered, %d pre-rendered during scan, %d failed).",
            rendered + concurrent_ready,
            rendered,
            concurrent_ready,
            failed,
        )
    else:
        logger.info("Done. %d rendered, %d failed.", rendered, failed)
    logger.info("Output: %s", highlights_dir)


if __name__ == "__main__":
    main()
