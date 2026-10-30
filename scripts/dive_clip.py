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

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.checkpoint import PipelineState  # noqa: E402
from utils.logging_setup import configure_logging, default_log_path  # noqa: E402

logger = logging.getLogger(__name__)


def run(cmd: list[str], description: str) -> int:
    """Run a command, streaming output to stdout/stderr. Returns exit code."""
    logger.info("=== %s ===", description)
    logger.debug("$ %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        logger.error("Command not found: %s", cmd[0])
        return 127
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130


def load_general_config(config_path: str | Path) -> dict:
    """Return the `general` section of a config file, or {} if unavailable."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("general") or {}


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path] | None:
    """Resolve input/output folders, CLI flags taking precedence over config.

    Returns None when neither source supplies the input folder.
    """
    general = load_general_config(args.config)

    input_value = args.input or general.get("input_dir")
    if not input_value:
        logger.error(
            "No input folder given. Pass --input or set general.input_dir in %s.",
            args.config,
        )
        return None

    output_value = args.output or general.get("output_dir")
    if not output_value:
        logger.error(
            "No output folder given. Pass --output or set general.output_dir in %s.",
            args.config,
        )
        return None

    return Path(input_value), Path(output_value)


def cmd_scan(args: argparse.Namespace) -> int:
    """Execute the full scan pipeline: ingest → scan → render.

    Supports checkpointing: if --resume is set, completed steps are skipped.
    State is stored in <output>/.pipeline_state.json.
    """
    resolved = resolve_paths(args)
    if resolved is None:
        return 2
    input_folder, output_folder = resolved
    config = args.config
    debug_mode = args.debug

    if not input_folder.exists():
        logger.error("Input folder does not exist: %s", input_folder)
        return 1

    output_folder.mkdir(parents=True, exist_ok=True)

    log_path = args.log_file or default_log_path(output_folder)
    configure_logging(debug=debug_mode, log_file=log_path)
    logger.info("Logging to %s", log_path)

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
        "--log-file",
        str(log_path),
    ]
    if debug_mode:
        ingest_cmd.append("--debug")

    if not state.is_step_done("ingest") or not manifest_path.exists():
        state.mark_step_started("ingest")
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
        "--output-dir",
        str(output_folder),
        "--state-path",
        str(state_path),
        "--log-file",
        str(log_path),
    ]
    if debug_mode:
        scan_cmd.append("--debug")

    if not state.is_step_done("scan") or not detections_path.exists():
        state.mark_step_started("scan")
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
        "--state-path",
        str(state_path),
        "--log-file",
        str(log_path),
    ]
    if debug_mode:
        render_cmd.append("--debug")

    if not state.is_step_done("render"):
        state.mark_step_started("render")
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

    logger.info("✓ Pipeline complete. Highlights in: %s", output_folder)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="dive-clip — Local footage curation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    scan_p = sub.add_parser("scan", help="Run ingest → scan → render pipeline")
    scan_p.add_argument(
        "--input",
        default=None,
        help="Path to footage folder (defaults to general.input_dir in the config)",
    )
    scan_p.add_argument(
        "--output",
        default=None,
        help="Output folder for highlights (defaults to general.output_dir in the config)",
    )
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
    scan_p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Logfile path (defaults to <output>/logs/run-<timestamp>.log)",
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
