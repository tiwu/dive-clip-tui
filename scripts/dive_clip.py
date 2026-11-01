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
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from rich.panel import Panel
from rich.rule import Rule

SCRIPTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.checkpoint import PipelineState  # noqa: E402
from utils.config import ConfigError, load_config, load_general_config  # noqa: E402
from utils.logging_setup import configure_logging, get_console  # noqa: E402
from utils.run_archive import (  # noqa: E402
    archive_run_artifacts,
    find_latest_run_dir,
    get_run_dir,
    resolve_assembly_artifacts,
    resolve_run_dir,
    update_latest_symlink,
)
from utils.signals import setup_signal_handlers  # noqa: E402
from utils.stats import (  # noqa: E402
    Phase0Stats,
    Phase1Stats,
    Phase2Stats,
    PipelineStats,
    RenderStats,
    format_stats_log,
    format_stats_table,
    load_stats_json,
    parse_duration,
    save_stats_json,
)
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
    try:
        general = load_general_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return None

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
    State is stored in <output>/runs/<run_id>/pipeline_state.json.
    """
    pipeline_start_time = time.perf_counter()
    try:
        general = load_general_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1
    output_value = args.output or general.get("output_dir") or "./outputs"
    output_folder = Path(output_value)
    output_folder.mkdir(parents=True, exist_ok=True)

    config = args.config
    debug_mode = args.debug

    resume_target = getattr(args, "resume", False)
    is_resuming = resume_target is not False

    if is_resuming:
        if isinstance(resume_target, str):
            try:
                run_dir = resolve_run_dir(output_folder, resume_target)
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                return 1
        else:
            latest_dir = find_latest_run_dir(output_folder)
            if latest_dir is not None:
                run_dir = latest_dir
            else:
                logger.info(
                    "No previous run found to resume in %s; starting a new run.", output_folder
                )
                run_dir = get_run_dir(output_folder)
    else:
        run_dir = get_run_dir(output_folder)

    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "pipeline_state.json"
    state = PipelineState(state_path)

    if args.reset:
        state.reset()
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir, ignore_errors=True)
        logger.info("Pipeline state reset.")

    # Resolve input folder (from CLI args, prior run metadata, or config)
    run_meta = state.get_run_metadata()
    input_value = (
        args.input
        or (run_meta.get("input_dir") if is_resuming else None)
        or general.get("input_dir")
    )
    if not input_value:
        logger.error(
            "No input folder given. Pass --input or set general.input_dir in %s.",
            args.config,
        )
        return 2

    input_folder = Path(input_value)
    if not input_folder.exists():
        logger.error("Input folder does not exist: %s", input_folder)
        return 1

    # Record run metadata in state
    state.init_run_metadata(
        run_id=run_dir.name,
        input_dir=input_folder,
        output_dir=output_folder,
        config_path=config,
        cli_args=sys.argv[1:],
    )

    version = get_version()
    console = get_console()
    if console.is_terminal:
        console.print(
            Panel(
                f"[bold white]Version:[/] [green]v{version}[/]\n"
                f"[bold white]Input:[/]   [yellow]{input_folder}[/]\n"
                f"[bold white]Output:[/]  [yellow]{output_folder}[/]\n"
                f"[bold white]Run:[/]     [cyan]{run_dir}[/]\n"
                f"[bold white]Config:[/]  [yellow]{config}[/]",
                title=f"[bold green]Dive Clip Curation Pipeline v{version}[/bold green]",
                border_style="green",
            )
        )

    log_path = args.log_file or (run_dir / "run.log")
    configure_logging(debug=debug_mode, log_file=log_path)
    update_latest_symlink(output_folder, run_dir)
    logger.info("Dive Clip Curation Pipeline v%s", version)
    logger.info("Logging to %s", log_path)

    manifest_path = run_dir / "manifest.json"
    legacy_manifest = run_dir / "manifest_raw.json"
    if not manifest_path.exists() and legacy_manifest.exists():
        manifest_path = legacy_manifest

    detections_path = run_dir / "detections_refined.json"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ingest
    should_assemble = bool(getattr(args, "assemble", False))
    total_steps = 4 if should_assemble else 3

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
            f"Step 1/{total_steps}: Ingest media (manifest)",
        )
        if rc != 0:
            if rc == 130:
                logger.warning("Pipeline interrupted by user.")
                return 130
            logger.error("Ingest failed with exit code %d. Aborting.", rc)
            return rc
        state.mark_step_done("ingest")
    else:
        logger.info("Step 1/%d: Ingest media (manifest) — SKIPPED (already done)", total_steps)

    # Step 2: Scan
    scan_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "02_scan_footage.py"),
        "--config",
        config,
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(run_dir),
        "--run-dir",
        str(run_dir),
        "--state-path",
        str(state_path),
        "--log-file",
        str(log_path),
    ]
    if debug_mode:
        scan_cmd.append("--debug")
    if is_resuming:
        scan_cmd.append("--resume")

    if not state.is_step_done("scan") or not detections_path.exists():
        state.mark_step_started("scan")
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
                f"Step 2/{total_steps}: Scan footage (AI detections)",
            )
        finally:
            stop_render_event.set()
            render_thread.join(timeout=2.0)

        if rc != 0:
            if rc == 130:
                logger.warning("Pipeline interrupted by user.")
                return 130
            logger.error("Scan failed with exit code %d. Aborting.", rc)
            return rc
        state.mark_step_done("scan")
    else:
        logger.info("Step 2/%d: Scan footage (AI detections) — SKIPPED (already done)", total_steps)

    # Step 3: Render
    render_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "05_render_clips.py"),
        "--config",
        config,
        "--input",
        str(detections_path),
        "--output",
        str(highlights_dir),
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
            f"Step 3/{total_steps}: Render clips (MP4 highlights)",
        )
        if rc != 0:
            if rc == 130:
                logger.warning("Pipeline interrupted by user.")
                return 130
            logger.error("Render failed with exit code %d. Aborting.", rc)
            return rc
        state.mark_step_done("render")
    else:
        logger.info(
            "Step 3/%d: Render clips (MP4 highlights) — SKIPPED (already done)", total_steps
        )

    # Step 4: Assemble (optional)
    assembled_video_path: Path | None = None
    if should_assemble:
        assemble_mode = getattr(args, "assemble_mode", None) or "aftermovie"
        top_n = _resolve_top_n(args, config)
        if assemble_mode == "simple":
            assembled_video_path = output_folder / "highlight_reel.mp4"
            assemble_script = SCRIPTS_DIR / "06_assemble_video.py"
            step_4_desc = f"Step 4/{total_steps}: Assemble highlight video"
        else:
            assembled_video_path = output_folder / "aftermovie.mp4"
            assemble_script = SCRIPTS_DIR / "07_create_aftermovie.py"
            step_4_desc = f"Step 4/{total_steps}: Assemble aftermovie"

        assemble_cmd = [
            sys.executable,
            str(assemble_script),
            "--config",
            config,
            "--input",
            str(detections_path),
            "--clips-dir",
            str(highlights_dir),
            "--output",
            str(assembled_video_path),
            "--top-n",
            str(top_n),
            "--log-file",
            str(log_path),
        ]
        music_arg = getattr(args, "music", None)
        if assemble_mode != "simple" and music_arg:
            assemble_cmd.extend(["--music", str(music_arg)])
        if debug_mode:
            assemble_cmd.append("--debug")

        if not state.is_step_done("assemble") or not assembled_video_path.exists():
            state.mark_step_started("assemble")
            rc = run(
                assemble_cmd,
                step_4_desc,
            )
            if rc != 0:
                if rc == 130:
                    logger.warning("Pipeline interrupted by user.")
                    return 130
                logger.error("Assemble failed with exit code %d. Aborting.", rc)
                return rc
            state.mark_step_done("assemble")
        else:
            logger.info("%s — SKIPPED (already done)", step_4_desc)

    pipeline_elapsed = time.perf_counter() - pipeline_start_time

    # Calculate footage stats from manifest
    video_count = 0
    total_video_duration = 0.0
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest_clips = manifest_data.get("clips", [])
            video_count = len(manifest_clips)
            total_video_duration = sum(parse_duration(c.get("duration")) for c in manifest_clips)
        except Exception as exc:
            logger.debug("Could not read manifest for stats: %s", exc)

    # Restore scan stats
    scan_data = state.get("scan_stats")
    p0_stats = Phase0Stats()
    p1_stats = Phase1Stats()
    p2_stats = Phase2Stats()
    if not scan_data and (run_dir / "run_stats.json").exists():
        try:
            prior_stats = load_stats_json(run_dir / "run_stats.json")
            p0_stats = prior_stats.phase0
            p1_stats = prior_stats.phase1
            p2_stats = prior_stats.phase2
        except Exception as exc:
            logger.debug("Could not restore prior scan stats from file: %s", exc)
    elif isinstance(scan_data, dict):
        if isinstance(scan_data.get("phase0"), dict):
            p0_stats = Phase0Stats.model_validate(scan_data["phase0"])
        if isinstance(scan_data.get("phase1"), dict):
            p1_stats = Phase1Stats.model_validate(scan_data["phase1"])
        if isinstance(scan_data.get("phase2"), dict):
            p2_stats = Phase2Stats.model_validate(scan_data["phase2"])

    # Restore or build render stats
    render_data = state.get("render_stats")
    if isinstance(render_data, dict):
        r_stats = RenderStats.model_validate(render_data)
    else:
        dets_list = []
        if detections_path.exists():
            try:
                with open(detections_path, encoding="utf-8") as f:
                    dets_list = json.load(f).get("detections", [])
            except Exception:
                pass
        rendered_files = list(highlights_dir.glob("*.mp4")) if highlights_dir.exists() else []
        total_highlight_dur = sum(
            max(0.0, float(d.get("end_time", 0)) - float(d.get("start_time", 0))) for d in dets_list
        )
        r_stats = RenderStats(
            clips_total=len(dets_list),
            clips_rendered=len(rendered_files),
            total_duration=round(total_highlight_dur, 2),
            elapsed_seconds=float(state.get("render_duration_seconds", 0.0) or 0.0),
        )

    step_durations = {}
    for step in ("ingest", "scan", "render", "assemble"):
        dur = state.get(f"{step}_duration_seconds")
        if dur is not None:
            step_durations[step] = float(dur)

    pipeline_stats = PipelineStats(
        run_id=run_dir.name,
        started_at=state.get("created_at", ""),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        total_elapsed_seconds=round(pipeline_elapsed, 2),
        video_count=video_count,
        total_video_duration=round(total_video_duration, 2),
        phase0=p0_stats,
        phase1=p1_stats,
        phase2=p2_stats,
        render=r_stats,
        step_durations=step_durations,
    )

    stats_path = run_dir / "run_stats.json"
    save_stats_json(pipeline_stats, stats_path)
    state.set("pipeline_stats", pipeline_stats.to_dict())
    state.save()

    archive_run_artifacts(
        run_dir,
        manifest_path=manifest_path,
        detections_path=detections_path,
        log_file=log_path,
        stats_path=stats_path,
        aftermovie_path=assembled_video_path,
    )
    update_latest_symlink(output_folder, run_dir)

    if console.is_terminal:
        console.print(format_stats_table(pipeline_stats, title="Dive Clip Pipeline Summary"))
        panel_lines = [
            "[bold green]Pipeline complete![/bold green]",
            f"[bold white]Highlights in:[/] [cyan]{highlights_dir}[/]",
        ]
        if should_assemble and assembled_video_path is not None:
            panel_lines.append(
                f"[bold white]Assembled video:[/] [bold cyan]{assembled_video_path}[/]"
            )
        panel_lines.extend([
            f"[bold white]Run archive:[/]   [cyan]{run_dir}[/]",
            f"[bold white]Run statistics:[/] [cyan]{stats_path}[/]",
        ])
        console.print(
            Panel(
                "\n".join(panel_lines),
                border_style="green",
            )
        )
    logger.info("✓ Pipeline complete. Highlights in: %s", highlights_dir)
    if should_assemble and assembled_video_path is not None:
        logger.info("✓ Final video assembled: %s", assembled_video_path)
    logger.info("Run artifacts in: %s", run_dir)
    logger.info(format_stats_log(pipeline_stats))
    return 0


def resolve_assembly_inputs(
    args: argparse.Namespace,
    default_filename: str,
) -> tuple[Path, Path, Path, Path | None] | None:
    """Resolve detections path, clips dir, output path, and latest run dir for assembly."""
    try:
        general = load_general_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return None

    output_dir = Path(general.get("output_dir") or "./outputs")
    raw_output = getattr(args, "output", None)
    if raw_output:
        output_path = Path(raw_output)
        if output_path.is_dir():
            output_dir = output_path
            output_path = output_dir / default_filename
    else:
        output_path = output_dir / default_filename

    detections_path, clips_dir, latest_run = resolve_assembly_artifacts(
        output_dir,
        input_detections=getattr(args, "input", None),
        clips_dir=getattr(args, "clips_dir", None),
    )

    return detections_path, clips_dir, output_path, latest_run


def _resolve_aftermovie_setting(
    args: argparse.Namespace,
    config_path: str | Path,
    attr_name: str,
    config_key: str,
    default_val: float,
) -> float:
    """Resolve an aftermovie float setting from CLI args or config file."""
    raw_val = getattr(args, attr_name, None)
    if raw_val is not None:
        return float(raw_val)
    try:
        full_cfg = load_config(config_path, default_to_empty=True)
        aftermovie_cfg = full_cfg.get("aftermovie")
        if isinstance(aftermovie_cfg, dict) and aftermovie_cfg.get(config_key) is not None:
            return float(aftermovie_cfg[config_key])
    except ConfigError:
        pass
    return default_val


def _resolve_top_n(args: argparse.Namespace, config_path: str | Path) -> int:
    """Resolve top_n from CLI args, falling back to aftermovie.top_n_clips in config."""
    raw_top_n = getattr(args, "top_n", None)
    if raw_top_n is not None:
        return int(raw_top_n)
    try:
        full_cfg = load_config(config_path, default_to_empty=True)
        aftermovie_cfg = full_cfg.get("aftermovie")
        if isinstance(aftermovie_cfg, dict) and aftermovie_cfg.get("top_n_clips") is not None:
            return int(aftermovie_cfg["top_n_clips"])
    except ConfigError:
        pass
    return 10


def cmd_assemble(args: argparse.Namespace) -> int:
    """Concatenate rendered highlight clips into a highlight reel."""
    resolved = resolve_assembly_inputs(args, "highlight_reel.mp4")
    if resolved is None:
        return 1
    detections_path, clips_dir, output_path, latest_run = resolved

    top_n = _resolve_top_n(args, args.config)
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "06_assemble_video.py"),
        "--config",
        args.config,
        "--input",
        str(detections_path),
        "--clips-dir",
        str(clips_dir),
        "--output",
        str(output_path),
        "--top-n",
        str(top_n),
    ]
    log_file = getattr(args, "log_file", None)
    if log_file:
        cmd.extend(["--log-file", str(log_file)])
    if getattr(args, "debug", False):
        cmd.append("--debug")

    rc = run(cmd, "Assemble highlight video")
    if rc == 0 and latest_run is not None:
        archive_run_artifacts(latest_run, aftermovie_path=output_path)
    return rc


def cmd_aftermovie(args: argparse.Namespace) -> int:
    """Create an aftermovie with cross-dissolve transitions and optional music."""
    resolved = resolve_assembly_inputs(args, "aftermovie.mp4")
    if resolved is None:
        return 1
    detections_path, clips_dir, output_path, latest_run = resolved

    top_n = _resolve_top_n(args, args.config)
    music_volume = _resolve_aftermovie_setting(
        args, args.config, "music_volume", "music_volume", 0.15
    )
    fade_duration = _resolve_aftermovie_setting(
        args, args.config, "fade_duration", "fade_duration", 1.0
    )
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "07_create_aftermovie.py"),
        "--config",
        args.config,
        "--input",
        str(detections_path),
        "--clips-dir",
        str(clips_dir),
        "--output",
        str(output_path),
        "--top-n",
        str(top_n),
        "--music-volume",
        str(music_volume),
        "--fade-duration",
        str(fade_duration),
    ]
    if getattr(args, "music", None):
        cmd.extend(["--music", str(args.music)])
    log_file = getattr(args, "log_file", None)
    if log_file:
        cmd.extend(["--log-file", str(log_file)])
    if getattr(args, "debug", False):
        cmd.append("--debug")

    rc = run(cmd, "Create aftermovie")
    if rc == 0 and latest_run is not None:
        archive_run_artifacts(latest_run, aftermovie_path=output_path)
    return rc


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
        nargs="?",
        const=True,
        default=False,
        metavar="RUN_ID_OR_PATH",
        help="Resume execution (defaults to latest run if no run ID or path specified)",
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
        help="Logfile path (defaults to <output>/runs/run-<timestamp>/run.log)",
    )
    scan_p.add_argument(
        "--assemble",
        action="store_true",
        default=False,
        help="Assemble rendered clips into a highlight video or aftermovie after scanning",
    )
    scan_p.add_argument(
        "--music",
        default=None,
        help="Path to background music track for aftermovie assembly",
    )
    scan_p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top-quality clips to include when assembling (default: 10, or config)",
    )
    scan_p.add_argument(
        "--assemble-mode",
        choices=["aftermovie", "simple"],
        default="aftermovie",
        help="Assembly mode when --assemble is enabled (default: aftermovie)",
    )
    scan_p.set_defaults(func=cmd_scan)

    assemble_p = sub.add_parser("assemble", help="Concatenate top-N rendered clips into a video")
    assemble_p.add_argument(
        "--input",
        default=None,
        help="Path to detections_refined.json (defaults to <output>/latest/detections_refined.json)",
    )
    assemble_p.add_argument(
        "--clips-dir",
        default=None,
        help="Path to rendered highlights (defaults to <output>/latest/highlights or <output>/highlights)",
    )
    assemble_p.add_argument(
        "--output",
        default=None,
        help="Path to output MP4 file (defaults to <output>/highlight_reel.mp4)",
    )
    assemble_p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top-quality clips to assemble (default: 10, or config)",
    )
    assemble_p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    assemble_p.add_argument("--debug", action="store_true", help="Enable debug logging")
    assemble_p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional logfile path",
    )
    assemble_p.set_defaults(func=cmd_assemble)

    aftermovie_p = sub.add_parser(
        "aftermovie", help="Create an aftermovie with crossfades and background music"
    )
    aftermovie_p.add_argument(
        "--input",
        default=None,
        help="Path to detections_refined.json (defaults to <output>/latest/detections_refined.json)",
    )
    aftermovie_p.add_argument(
        "--clips-dir",
        default=None,
        help="Path to rendered highlights (defaults to <output>/latest/highlights or <output>/highlights)",
    )
    aftermovie_p.add_argument(
        "--output",
        default=None,
        help="Path to output MP4 file (defaults to <output>/aftermovie.mp4)",
    )
    aftermovie_p.add_argument(
        "--music",
        default=None,
        help="Path to background music track (optional)",
    )
    aftermovie_p.add_argument(
        "--music-volume",
        type=float,
        default=None,
        help="Music ducking level (default: 0.15, or config)",
    )
    aftermovie_p.add_argument(
        "--fade-duration",
        type=float,
        default=None,
        help="Transition crossfade duration in seconds (default: 1.0, or config)",
    )
    aftermovie_p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top-quality clips to include (default: 10, or config)",
    )
    aftermovie_p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    aftermovie_p.add_argument("--debug", action="store_true", help="Enable debug logging")
    aftermovie_p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional logfile path",
    )
    aftermovie_p.set_defaults(func=cmd_aftermovie)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the dive-clip console script."""
    setup_signal_handlers()
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(debug=getattr(args, "debug", False))
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
