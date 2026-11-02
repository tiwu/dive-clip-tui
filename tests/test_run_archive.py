"""Tests for utils/run_archive.py."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.run_archive import (
    archive_file,
    archive_run_artifacts,
    default_run_log_path,
    generate_run_id,
    get_run_dir,
    update_latest_symlink,
)


def test_generate_run_id_format() -> None:
    """Run ID follows run-YYYYMMDD-HHMMSS format or uses provided timestamp."""
    run_id = generate_run_id()
    assert re.fullmatch(r"run-\d{8}-\d{6}", run_id)

    custom_id = generate_run_id("20261031-120000")
    assert custom_id == "run-20261031-120000"


def test_get_run_dir(tmp_path: Path) -> None:
    """Run directory path is under <output>/runs/<run_id>."""
    run_dir = get_run_dir(tmp_path, "run-20261031-120000")
    assert run_dir == tmp_path / "runs" / "run-20261031-120000"


def test_default_run_log_path(tmp_path: Path) -> None:
    """Default run log lives directly inside the run directory."""
    log_path = default_run_log_path(tmp_path, "run-20261031-120000")
    assert log_path == tmp_path / "runs" / "run-20261031-120000" / "run.log"


def test_archive_file_copies_when_exists(tmp_path: Path) -> None:
    """archive_file copies file to target dir and returns destination path."""
    src = tmp_path / "sample.json"
    src.write_text('{"hello": "world"}', encoding="utf-8")

    dest_dir = tmp_path / "archive"
    archived = archive_file(src, dest_dir)

    assert archived is not None
    assert archived == dest_dir / "sample.json"
    assert archived.read_text(encoding="utf-8") == '{"hello": "world"}'


def test_archive_file_with_target_name(tmp_path: Path) -> None:
    """archive_file supports custom target name."""
    src = tmp_path / "original.txt"
    src.write_text("test data", encoding="utf-8")

    dest_dir = tmp_path / "archive"
    archived = archive_file(src, dest_dir, target_name="renamed.txt")

    assert archived == dest_dir / "renamed.txt"
    assert archived.read_text(encoding="utf-8") == "test data"


def test_archive_file_returns_none_when_source_missing(tmp_path: Path) -> None:
    """archive_file returns None when source file does not exist."""
    src = tmp_path / "nonexistent.json"
    dest_dir = tmp_path / "archive"
    assert archive_file(src, dest_dir) is None


def test_archive_run_artifacts_all(tmp_path: Path) -> None:
    """archive_run_artifacts copies manifest, detections, and log to run_dir."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    manifest = output_dir / "manifest_raw.json"
    manifest.write_text('{"manifest": true}', encoding="utf-8")

    detections = output_dir / "detections_refined.json"
    detections.write_text('{"detections": []}', encoding="utf-8")

    log_file = tmp_path / "outside.log"
    log_file.write_text("log line", encoding="utf-8")

    stats_file = output_dir / "run_stats.json"
    stats_file.write_text('{"video_count": 1}', encoding="utf-8")

    run_dir = output_dir / "runs" / "run-20261031-120000"
    archived = archive_run_artifacts(
        run_dir,
        manifest_path=manifest,
        detections_path=detections,
        log_file=log_file,
        stats_path=stats_file,
    )

    assert (run_dir / "manifest_raw.json").exists()
    assert (run_dir / "detections_refined.json").exists()
    assert (run_dir / "run_stats.json").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "run.log").read_text(encoding="utf-8") == "log line"
    assert archived["manifest"] == run_dir / "manifest_raw.json"
    assert archived["detections"] == run_dir / "detections_refined.json"
    assert archived["stats"] == run_dir / "run_stats.json"
    assert archived["log"] == run_dir / "run.log"


def test_archive_run_artifacts_handles_already_in_place_log(tmp_path: Path) -> None:
    """archive_run_artifacts does not copy log when it already lives at run_dir/run.log."""
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    in_place_log = run_dir / "run.log"
    in_place_log.write_text("existing log", encoding="utf-8")

    archived = archive_run_artifacts(run_dir, log_file=in_place_log)
    assert archived["log"] == in_place_log
    assert in_place_log.read_text(encoding="utf-8") == "existing log"


def test_update_latest_symlink_exfat_fallback(tmp_path: Path) -> None:
    """update_latest_symlink should write latest.txt when symlink_to raises OSError."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_dir = out_dir / "runs" / "run-20261101-100000"
    run_dir.mkdir(parents=True)

    with patch.object(Path, "symlink_to", side_effect=OSError(1, "Operation not permitted")):
        update_latest_symlink(out_dir, run_dir)

    fallback_file = out_dir / "latest.txt"
    assert fallback_file.exists()
    assert "run-20261101-100000" in fallback_file.read_text(encoding="utf-8")


def test_archive_run_artifacts_with_aftermovie(tmp_path: Path) -> None:
    """archive_run_artifacts copies aftermovie_path into run_dir when present."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    aftermovie = output_dir / "aftermovie.mp4"
    aftermovie.write_bytes(b"fake mp4 data")

    run_dir = output_dir / "runs" / "run-20261101-120000"
    archived = archive_run_artifacts(
        run_dir,
        aftermovie_path=aftermovie,
    )

    assert "aftermovie" in archived
    assert archived["aftermovie"] == run_dir / "aftermovie.mp4"
    assert (run_dir / "aftermovie.mp4").read_bytes() == b"fake mp4 data"

    # Missing aftermovie file is safely ignored
    missing_run_dir = output_dir / "runs" / "run-20261101-130000"
    archived_missing = archive_run_artifacts(
        missing_run_dir,
        aftermovie_path=output_dir / "nonexistent.mp4",
    )
    assert "aftermovie" not in archived_missing


def test_archive_run_artifacts_with_classifications(tmp_path: Path) -> None:
    """archive_run_artifacts copies classifications_path into run_dir as classifications.json."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    classifications = output_dir / "custom_classifications.json"
    classifications.write_text('{"total_clips": 1, "classifications": []}', encoding="utf-8")

    run_dir = output_dir / "runs" / "run-20261102-120000"
    archived = archive_run_artifacts(
        run_dir,
        classifications_path=classifications,
    )

    assert "classifications" in archived
    assert archived["classifications"] == run_dir / "classifications.json"
    assert (run_dir / "classifications.json").exists()
    assert "total_clips" in (run_dir / "classifications.json").read_text(encoding="utf-8")


def test_resolve_assembly_artifacts(tmp_path: Path) -> None:
    """resolve_assembly_artifacts resolves from explicit paths, latest run, or output dir."""
    from utils.run_archive import resolve_assembly_artifacts, update_latest_symlink

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # 1. Fallback to output_dir when no latest run exists
    dets, clips, run_dir = resolve_assembly_artifacts(output_dir)
    assert dets == output_dir / "detections_refined.json"
    assert clips == output_dir / "highlights"
    assert run_dir is None

    # 2. Automatically resolves from latest run when artifacts exist
    run_1 = output_dir / "runs" / "run-20261101-100000"
    run_1_highlights = run_1 / "highlights"
    run_1_highlights.mkdir(parents=True)
    run_1_dets = run_1 / "detections_refined.json"
    run_1_dets.write_text("{}", encoding="utf-8")
    update_latest_symlink(output_dir, run_1)

    dets, clips, run_dir = resolve_assembly_artifacts(output_dir)
    assert dets == run_1_dets
    assert clips == run_1_highlights
    assert run_dir == run_1

    # 3. Explicit paths override run archive
    custom_dets = tmp_path / "custom_dets.json"
    custom_clips = tmp_path / "custom_clips"
    dets, clips, run_dir = resolve_assembly_artifacts(
        output_dir, input_detections=custom_dets, clips_dir=custom_clips
    )
    assert dets == custom_dets
    assert clips == custom_clips
    assert run_dir == run_1


def test_resolve_and_validate_assembly_artifacts_success(tmp_path: Path) -> None:
    """resolve_and_validate_assembly_artifacts succeeds when files exist."""
    from utils.run_archive import resolve_and_validate_assembly_artifacts

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dets_file = output_dir / "detections_refined.json"
    dets_file.write_text("{}", encoding="utf-8")
    clips_dir = output_dir / "highlights"
    clips_dir.mkdir()

    resolved_dets, resolved_clips, _run_dir = resolve_and_validate_assembly_artifacts(output_dir)
    assert resolved_dets == dets_file
    assert resolved_clips == clips_dir


def test_resolve_and_validate_assembly_artifacts_missing_detections(tmp_path: Path) -> None:
    """resolve_and_validate_assembly_artifacts raises FileNotFoundError when detections missing."""
    from utils.run_archive import resolve_and_validate_assembly_artifacts

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    clips_dir = output_dir / "highlights"
    clips_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"detections_refined\.json not found"):
        resolve_and_validate_assembly_artifacts(output_dir)


def test_resolve_and_validate_assembly_artifacts_missing_clips(tmp_path: Path) -> None:
    """resolve_and_validate_assembly_artifacts raises FileNotFoundError when clips dir missing."""
    from utils.run_archive import resolve_and_validate_assembly_artifacts

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dets_file = output_dir / "detections_refined.json"
    dets_file.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Clips directory not found"):
        resolve_and_validate_assembly_artifacts(output_dir)
