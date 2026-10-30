"""dive-clip CLI — single entry point for the local footage curation pipeline.

Usage:
    dive-clip scan --input ./footage --output ./highlights

Chains: ingest (manifest) → scan (AI detections) → render (MP4 clips).
This is Phase 1 Step 1.4 of the simple plan.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from rich.panel import Panel
from rich.rule import Rule

SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.checkpoint import PipelineState  # noqa: E402
from utils.logging_setup import configure_logging, default_log_path, get_console  # noqa: E402
from utils.version import get_version  # noqa: E402

logger = logging.getLogger(__name__)


def run(cmd: list[str], description: str) -> int:
    """Run a command, streaming output to stdout/stderr. Returns exit code."""
    console = get_console()
    if console.is_terminal:
        console.print(Rule(f"[bold cyan]{description}[/bold cyan]", style="cyan"))
    else:
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


def _run_background_render_worker(
    detections_path: Path,
    highlights_dir: Path,
    stop_event: threading.Event,
) -> None:
    """Periodically inspect detections_refined.json and render newly detected clips."""
    try:
        render_spec = importlib.util.spec_from_file_location(
            "scripts_05_render_clips", SCRIPTS_DIR / "05_render_clips.py"
        )
        if render_spec and render_spec.loader:
            render_mod = importlib.util.module_from_spec(render_spec)
            render_spec.loader.exec_module(render_mod)
            render_detections = getattr(render_mod, "render_detections", None)
        else:
            return
    except Exception as exc:
        logger.debug("Could not initialize background renderer: %s", exc)
        return

    if render_detections is None:
        return

    while not stop_event.is_set():
        if detections_path.exists():
            try:
                with open(detections_path, encoding="utf-8") as f:
                    data = json.load(f)
                dets = data.get("detections", [])
                if dets:
                    render_detections(dets, highlights_dir, log_to_console=False)
            except Exception as exc:
                logger.debug("Background render check: %s", exc)
        stop_event.wait(1.0)


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

    version = get_version()
    console = get_console()
    if console.is_terminal:
        console.print(
            Panel(
                f"[bold white]Version:[/] [green]v{version}[/]\n"
                f"[bold white]Input:[/]   [yellow]{input_folder}[/]\n"
                f"[bold white]Output:[/]  [yellow]{output_folder}[/]\n"
                f"[bold white]Config:[/]  [yellow]{config}[/]",
                title=f"[bold green]Dive Clip Curation Pipeline v{version}[/bold green]",
                border_style="green",
            )
        )

    log_path = args.log_file or default_log_path(output_folder)
    configure_logging(debug=debug_mode, log_file=log_path)
    logger.info("Dive Clip Curation Pipeline v%s", version)
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
    if getattr(args, "resume", False):
        scan_cmd.append("--resume")

    if not state.is_step_done("scan") or not detections_path.exists():
        state.mark_step_started("scan")
        highlights_dir = output_folder / "highlights"
        stop_render_event = threading.Event()
        render_thread = threading.Thread(
            target=_run_background_render_worker,
            args=(detections_path, highlights_dir, stop_render_event),
            daemon=True,
        )
        render_thread.start()
        try:
            rc = run(
                scan_cmd,
                "Step 2/3: Scan footage (AI detections)",
            )
        finally:
            stop_render_event.set()
            render_thread.join(timeout=2.0)

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

    if console.is_terminal:
        console.print(
            Panel(
                f"[bold green]Pipeline complete![/bold green]\n"
                f"[bold white]Highlights in:[/] [cyan]{output_folder}[/]",
                border_style="green",
            )
        )
    logger.info("✓ Pipeline complete. Highlights in: %s", output_folder)
    return 0


def build_parser() -> argparse.ArgumentParser:
    version = get_version()
    parser = argparse.ArgumentParser(description="dive-clip — Local footage curation pipeline")
    parser.add_argument(
        "--version",
        action="version",
        version=f"dive-clip v{version}",
    )
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
