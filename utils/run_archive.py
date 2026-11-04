"""Run archiving utilities for pipeline artifacts.

Provides functions to organize runs into timestamped directories under
`<output_dir>/runs/run-<timestamp>/` containing execution logs, raw manifests,
and refined detections.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_run_id(timestamp: str | None = None) -> str:
    """Return a run identifier formatted as `run-<timestamp>`.

    Defaults to the current local timestamp formatted as YYYYMMDD-HHMMSS.
    """
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}"


def get_run_dir(output_dir: Path | str, run_id: str | None = None) -> Path:
    """Return the run directory path under `<output_dir>/runs/<run_id>/`."""
    resolved_id = run_id or generate_run_id()
    return Path(output_dir) / "runs" / resolved_id


def find_latest_run_dir(output_dir: Path | str) -> Path | None:
    """Return the most recent run directory under `<output_dir>/runs/`, or None."""
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def resolve_run_dir(output_dir: Path | str, target: str | Path | None = None) -> Path:
    """Resolve an existing run directory by run ID or explicit path.

    If target is None, returns a new timestamped run directory.
    If target is specified but not found, raises FileNotFoundError.
    """
    if target is None:
        return get_run_dir(output_dir)

    direct_path = Path(target)
    if direct_path.is_dir():
        return direct_path

    relative_run = Path(output_dir) / "runs" / target
    if relative_run.is_dir():
        return relative_run

    raise FileNotFoundError(f"Run directory not found: {target}")


def update_latest_symlink(output_dir: Path | str, run_dir: Path | str) -> Path:
    """Point `<output_dir>/latest` symlink to `run_dir` using a relative target."""
    out_path = Path(output_dir)
    target_path = Path(run_dir).resolve()
    latest_link = out_path / "latest"

    try:
        rel_target = target_path.relative_to(out_path.resolve())
    except ValueError:
        rel_target = target_path

    if latest_link.is_symlink() or latest_link.exists():
        try:
            latest_link.unlink()
        except OSError as exc:
            logger.debug("Could not remove existing symlink %s: %s", latest_link, exc)

    try:
        latest_link.symlink_to(rel_target)
        logger.debug("Updated latest symlink: %s -> %s", latest_link, rel_target)
    except OSError as exc:
        fallback_txt = out_path / "latest.txt"
        try:
            fallback_txt.write_text(f"{rel_target}\n", encoding="utf-8")
            logger.debug(
                "Filesystem does not support symlinks (%s); wrote fallback latest.txt: %s",
                exc,
                rel_target,
            )
        except OSError as write_exc:
            logger.warning(
                "Could not create latest symlink or fallback latest.txt at %s: %s (symlink error: %s)",
                fallback_txt,
                write_exc,
                exc,
            )

    return latest_link


def default_run_log_path(output_dir: Path | str, run_id: str | None = None) -> Path:
    """Return the standard log path inside the run directory."""
    return get_run_dir(output_dir, run_id=run_id) / "run.log"


def resolve_assembly_artifacts(
    output_dir: Path | str,
    input_detections: Path | str | None = None,
    clips_dir: Path | str | None = None,
) -> tuple[Path, Path, Path | None]:
    """Resolve detections JSON path, clips directory, and latest run dir for assembly.

    Searches explicit paths, the latest symlink, recent timestamped runs, and output_dir.
    Returns (detections_path, clips_dir, latest_run).
    """
    out_path = Path(output_dir)
    latest_link = out_path / "latest"
    latest_run: Path | None = (
        latest_link.resolve() if latest_link.exists() else find_latest_run_dir(out_path)
    )

    if input_detections:
        detections_path = Path(input_detections)
    elif (latest_link / "detections_refined.json").exists():
        detections_path = (latest_link / "detections_refined.json").resolve()
    elif latest_run and (latest_run / "detections_refined.json").exists():
        detections_path = latest_run / "detections_refined.json"
    else:
        detections_path = out_path / "detections_refined.json"

    if clips_dir:
        resolved_clips_dir = Path(clips_dir)
    elif (detections_path.parent / "highlights").exists():
        resolved_clips_dir = detections_path.parent / "highlights"
    elif (latest_link / "highlights").exists():
        resolved_clips_dir = (latest_link / "highlights").resolve()
    elif latest_run and (latest_run / "highlights").exists():
        resolved_clips_dir = latest_run / "highlights"
    else:
        resolved_clips_dir = out_path / "highlights"

    return detections_path, resolved_clips_dir, latest_run


def validate_assembly_inputs(
    detections_path: Path,
    clips_dir: Path,
) -> Path:
    """Validate that detections JSON and clips directory exist, falling back to repo outputs.

    Returns the verified detections Path, or raises FileNotFoundError.
    """
    if not detections_path.exists():
        alt = Path(__file__).resolve().parent.parent / "outputs" / "detections_refined.json"
        if alt.exists():
            detections_path = alt
        else:
            raise FileNotFoundError(
                f"detections_refined.json not found at {detections_path}. Run 02_scan_footage.py first."
            )

    if not clips_dir.exists():
        raise FileNotFoundError(
            f"Clips directory not found: {clips_dir}. Run 05_render_clips.py first to render detection clips."
        )

    return detections_path


def resolve_and_validate_assembly_artifacts(
    output_dir: Path | str,
    input_detections: Path | str | None = None,
    clips_dir: Path | str | None = None,
) -> tuple[Path, Path, Path | None]:
    """Resolve and validate detections path and clips directory for video assembly.

    Returns (verified_detections_path, clips_dir, latest_run).
    Raises FileNotFoundError if artifacts are missing.
    """
    detections_path, resolved_clips_dir, latest_run = resolve_assembly_artifacts(
        output_dir, input_detections=input_detections, clips_dir=clips_dir
    )
    detections_path = validate_assembly_inputs(detections_path, resolved_clips_dir)
    return detections_path, resolved_clips_dir, latest_run


def archive_file(
    source: Path | str,
    destination_dir: Path | str,
    target_name: str | None = None,
) -> Path | None:
    """Copy `source` to `destination_dir / (target_name or source.name)` if source exists.

    Returns the destination Path, or None if the source does not exist.
    """
    src = Path(source)
    if not src.exists():
        return None
    dest_dir = Path(destination_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / (target_name or src.name)
    if src.resolve() == dest_file.resolve():
        return dest_file
    shutil.copy2(src, dest_file)
    logger.debug("Archived %s -> %s", src, dest_file)
    return dest_file


def archive_run_artifacts(
    run_dir: Path | str,
    manifest_path: Path | str | None = None,
    detections_path: Path | str | None = None,
    log_file: Path | str | None = None,
    stats_path: Path | str | None = None,
    aftermovie_path: Path | str | None = None,
    classifications_path: Path | str | None = None,
    pass1_coarse_trace_path: Path | str | None = None,
    pass2_fine_trace_path: Path | str | None = None,
    selection_trace_path: Path | str | None = None,
    prescreen_trace_path: Path | str | None = None,
) -> dict[str, Path]:
    """Archive available pipeline artifacts into `run_dir`.

    Returns a dictionary of artifact keys to their archived paths.
    """
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    archived: dict[str, Path] = {}

    if manifest_path is not None:
        src_manifest = Path(manifest_path)
        dest = archive_file(src_manifest, target_dir, src_manifest.name)
        if dest:
            archived["manifest"] = dest

    if detections_path is not None:
        dest = archive_file(detections_path, target_dir, "detections_refined.json")
        if dest:
            archived["detections"] = dest

    if classifications_path is not None:
        src_classifications = Path(classifications_path)
        dest = archive_file(src_classifications, target_dir, "classifications.json")
        if dest:
            archived["classifications"] = dest

    if pass1_coarse_trace_path is not None:
        src_p1 = Path(pass1_coarse_trace_path)
        dest = archive_file(src_p1, target_dir, "pass1_coarse_trace.json")
        if dest:
            archived["pass1_coarse_trace"] = dest

    if pass2_fine_trace_path is not None:
        src_p2 = Path(pass2_fine_trace_path)
        dest = archive_file(src_p2, target_dir, "pass2_fine_trace.json")
        if dest:
            archived["pass2_fine_trace"] = dest

    if selection_trace_path is not None:
        src_sel = Path(selection_trace_path)
        dest = archive_file(src_sel, target_dir, "selection_trace.json")
        if dest:
            archived["selection_trace"] = dest

    if prescreen_trace_path is not None:
        src_ps = Path(prescreen_trace_path)
        dest = archive_file(src_ps, target_dir, "prescreen_trace.json")
        if dest:
            archived["prescreen_trace"] = dest

    if stats_path is not None:
        dest = archive_file(stats_path, target_dir, "run_stats.json")
        if dest:
            archived["stats"] = dest

    if aftermovie_path is not None:
        src_aftermovie = Path(aftermovie_path)
        dest = archive_file(src_aftermovie, target_dir, src_aftermovie.name)
        if dest:
            archived["aftermovie"] = dest

    if log_file is not None:
        src_log = Path(log_file)
        dest_log = target_dir / "run.log"
        # Only copy if source exists and is not already the target file
        if src_log.exists():
            if src_log.resolve() != dest_log.resolve():
                dest = archive_file(src_log, target_dir, "run.log")
                if dest:
                    archived["log"] = dest
            else:
                archived["log"] = src_log

    return archived
