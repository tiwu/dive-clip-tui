"""dive-clip CLI — single entry point for the local footage curation pipeline.

Usage:
    dive-clip scan --input ./footage --output ./highlights

Chains: ingest (manifest) → scan (AI detections) → render (MP4 clips).
This is Phase 1 Step 1.4 of the simple plan.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.checkpoint import PipelineState  # noqa: E402

logger = logging.getLogger(__name__)


def run(cmd: list[str], description: str) -> int:
    """Run a command, streaming output to stdout/stderr. Returns exit code."""
    print(f"\n=== {description} ===")
    print(f"$ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print(f"ERROR: Command not found: {cmd[0]}")
        return 127
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130


def cmd_scan(args: argparse.Namespace) -> int:
    """Execute the full scan pipeline: ingest → scan → render.

    Supports checkpointing: if --resume is set, completed steps are skipped.
    State is stored in <output>/.pipeline_state.json.
    """
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    config = args.config
    debug_mode = args.debug

    if not input_folder.exists():
        logger.error("Input folder does not exist: %s", input_folder)
        return 1

    output_folder.mkdir(parents=True, exist_ok=True)

    # Configure logging based on debug flag
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    state_path = output_folder / ".pipeline_state.json"
    state = PipelineState(state_path)

    if args.reset:
        state.reset()
        logger.info("Pipeline state reset.")

    manifest_path = output_folder / "manifest_raw.json"
    detections_path = output_folder / "detections_refined.json"

    # Step 1: Ingest
    ingest_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "00_ingest_media.py"),
        "--input",
        str(input_folder),
        "--output",
        str(manifest_path),
    ]
    if debug_mode:
        ingest_cmd.append("--debug")

    if not state.is_step_done("ingest") or not manifest_path.exists():
        rc = run(
            ingest_cmd,
            "Step 1/3: Ingest media (manifest)",
        )
        if rc != 0:
            logger.error("Ingest failed with exit code %d. Aborting.", rc)
            return rc
        state.mark_step_done("ingest")
    else:
        logger.info("Step 1/3: Ingest media (manifest) — SKIPPED (already done)")

    # Step 2: Scan
    scan_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "02_scan_footage.py"),
        "--config",
        config,
    ]
    if debug_mode:
        scan_cmd.append("--debug")

    if not state.is_step_done("scan") or not detections_path.exists():
        rc = run(
            scan_cmd,
            "Step 2/3: Scan footage (AI detections)",
        )
        if rc != 0:
            logger.error("Scan failed with exit code %d. Aborting.", rc)
            return rc
        state.mark_step_done("scan")
    else:
        logger.info("Step 2/3: Scan footage (AI detections) — SKIPPED (already done)")

    # Step 3: Render
    render_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "05_render_clips.py"),
        "--config",
        config,
        "--input",
        str(detections_path),
        "--output",
        str(output_folder),
    ]
    if debug_mode:
        render_cmd.append("--debug")

    if not state.is_step_done("render"):
        rc = run(
            render_cmd,
            "Step 3/3: Render clips (MP4 highlights)",
        )
        if rc != 0:
            logger.error("Render failed with exit code %d. Aborting.", rc)
            return rc
        state.mark_step_done("render")
    else:
        logger.info("Step 3/3: Render clips (MP4 highlights) — SKIPPED (already done)")

    print(f"\n✓ Pipeline complete. Highlights in: {output_folder}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dive-clip — Local footage curation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    scan_p = sub.add_parser("scan", help="Run ingest → scan → render pipeline")
    scan_p.add_argument("--input", required=True, help="Path to footage folder")
    scan_p.add_argument("--output", required=True, help="Output folder for highlights")
    scan_p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    scan_p.add_argument("--debug", action="store_true", help="Enable debug logging")
    scan_p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed checkpoint",
    )
    scan_p.add_argument(
        "--reset",
        action="store_true",
        help="Reset pipeline state before starting",
    )
    scan_p.set_defaults(func=cmd_scan)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the dive-clip console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
