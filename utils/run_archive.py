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


def default_run_log_path(output_dir: Path | str, run_id: str | None = None) -> Path:
    """Return the standard log path inside the run directory."""
    return get_run_dir(output_dir, run_id=run_id) / "run.log"


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
    shutil.copy2(src, dest_file)
    logger.debug("Archived %s -> %s", src, dest_file)
    return dest_file


def archive_run_artifacts(
    run_dir: Path | str,
    manifest_path: Path | str | None = None,
    detections_path: Path | str | None = None,
    log_file: Path | str | None = None,
) -> dict[str, Path]:
    """Archive available pipeline artifacts into `run_dir`.

    Returns a dictionary of artifact keys to their archived paths.
    """
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    archived: dict[str, Path] = {}

    if manifest_path is not None:
        dest = archive_file(manifest_path, target_dir, "manifest_raw.json")
        if dest:
            archived["manifest"] = dest

    if detections_path is not None:
        dest = archive_file(detections_path, target_dir, "detections_refined.json")
        if dest:
            archived["detections"] = dest

    if log_file is not None:
        src_log = Path(log_file)
        dest_log = target_dir / "run.log"
        # Only copy if source exists and is not already the target file
        if src_log.resolve() != dest_log.resolve():
            dest = archive_file(src_log, target_dir, "run.log")
            if dest:
                archived["log"] = dest
        elif src_log.exists():
            archived["log"] = src_log

    return archived
