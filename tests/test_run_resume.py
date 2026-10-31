"""Tests for self-contained run directories, artifact renaming, and resume support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dive_clip import build_parser, cmd_scan
from utils.checkpoint import PipelineState
from utils.run_archive import (
    find_latest_run_dir,
    resolve_run_dir,
    update_latest_symlink,
)


def test_pipeline_state_run_metadata(tmp_path: Path) -> None:
    """PipelineState should record and reload run context metadata."""
    state_path = tmp_path / "pipeline_state.json"
    state = PipelineState(state_path)

    state.init_run_metadata(
        run_id="run-20261031-120000",
        input_dir="/path/to/footage",
        output_dir="/path/to/output",
        config_path="config.yaml",
        cli_args=["scan", "--input", "/path/to/footage"],
    )

    meta = state.get_run_metadata()
    assert meta["run_id"] == "run-20261031-120000"
    assert meta["input_dir"] == "/path/to/footage"
    assert meta["output_dir"] == "/path/to/output"
    assert meta["config_path"] == "config.yaml"
    assert "created_at" in meta

    # Verify reloading from disk
    reloaded = PipelineState(state_path)
    reloaded_meta = reloaded.get_run_metadata()
    assert reloaded_meta["run_id"] == "run-20261031-120000"
    assert reloaded_meta["input_dir"] == "/path/to/footage"


def test_find_latest_run_dir(tmp_path: Path) -> None:
    """find_latest_run_dir should find the newest timestamped run dir."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    assert find_latest_run_dir(tmp_path) is None

    run1 = runs_dir / "run-20261031-100000"
    run2 = runs_dir / "run-20261031-120000"
    run1.mkdir()
    run2.mkdir()

    latest = find_latest_run_dir(tmp_path)
    assert latest == run2


def test_resolve_run_dir_by_id_or_path(tmp_path: Path) -> None:
    """resolve_run_dir should resolve run IDs, explicit paths, and new IDs."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    existing_run = runs_dir / "run-20261031-110000"
    existing_run.mkdir()

    # Resolve by ID name under runs/
    resolved = resolve_run_dir(tmp_path, "run-20261031-110000")
    assert resolved == existing_run

    # Resolve by exact path
    resolved_path = resolve_run_dir(tmp_path, str(existing_run))
    assert resolved_path == existing_run

    # Nonexistent run target raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        resolve_run_dir(tmp_path, "run-nonexistent")


def test_update_latest_symlink(tmp_path: Path) -> None:
    """update_latest_symlink should create or update <output>/latest symlink."""
    runs_dir = tmp_path / "runs"
    run1 = runs_dir / "run-20261031-100000"
    run2 = runs_dir / "run-20261031-120000"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)

    symlink_path = update_latest_symlink(tmp_path, run1)
    assert symlink_path.is_symlink()
    assert symlink_path.resolve() == run1.resolve()

    # Update to run2
    update_latest_symlink(tmp_path, run2)
    assert symlink_path.resolve() == run2.resolve()


def test_parser_resume_overload() -> None:
    """build_parser should accept --resume without arg or with target run."""
    parser = build_parser()

    # Without --resume
    args = parser.parse_args(["scan", "--input", "in", "--output", "out"])
    assert args.resume is False

    # --resume with no value -> True
    args_bare = parser.parse_args(["scan", "--input", "in", "--output", "out", "--resume"])
    assert args_bare.resume is True

    # --resume with specific run target -> string
    args_target = parser.parse_args([
        "scan",
        "--input",
        "in",
        "--output",
        "out",
        "--resume",
        "run-20261031-100000",
    ])
    assert args_target.resume == "run-20261031-100000"


def test_cmd_scan_creates_self_contained_run_workspace(tmp_path: Path) -> None:
    """cmd_scan should place state, manifest.json, detections, log, and highlights inside run_dir."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        str(input_folder),
        "--output",
        str(output_folder),
        "--config",
        "config.yaml",
    ])

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> MagicMock:
        script_name = cmd[1]
        # Ingest step writes manifest.json to run_dir
        if "00_ingest_media.py" in script_name:
            out_idx = cmd.index("--output") + 1
            Path(cmd[out_idx]).write_text('{"files": []}')
        # Scan step writes detections_refined.json to run_dir
        elif "02_scan_footage.py" in script_name:
            run_dir_idx = cmd.index("--run-dir") + 1
            run_dir_path = Path(cmd[run_dir_idx])
            (run_dir_path / "detections_refined.json").write_text('{"detections": []}')
        return MagicMock(returncode=0)

    with patch("scripts.dive_clip.subprocess.run", side_effect=fake_subprocess_run):
        rc = cmd_scan(args)

    assert rc == 0

    runs = list((output_folder / "runs").glob("run-*"))
    assert len(runs) == 1
    run_dir = runs[0]

    # Verify self-contained files inside run_dir
    assert (run_dir / "pipeline_state.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "detections_refined.json").exists()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "highlights").is_dir()

    # Verify latest symlink
    latest = output_folder / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()

    # Verify metadata in pipeline_state.json
    state = PipelineState(run_dir / "pipeline_state.json")
    meta = state.get_run_metadata()
    assert meta["input_dir"] == str(input_folder)
    assert meta["output_dir"] == str(output_folder)
    assert state.is_step_done("ingest")
    assert state.is_step_done("scan")
    assert state.is_step_done("render")


def test_cmd_scan_resume_latest_and_restores_params(tmp_path: Path) -> None:
    """cmd_scan --resume should pick up latest run and restore omitted input folder."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    # Simulate an interrupted run
    run_dir = output_folder / "runs" / "run-20261031-100000"
    run_dir.mkdir(parents=True)
    state = PipelineState(run_dir / "pipeline_state.json")
    state.init_run_metadata(
        run_id="run-20261031-100000",
        input_dir=str(input_folder),
        output_dir=str(output_folder),
        config_path="config.yaml",
    )
    # Mark ingest done, scan not done
    state.mark_step_done("ingest")
    (run_dir / "manifest.json").write_text('{"files": []}')

    # Update latest symlink to point to this run
    update_latest_symlink(output_folder, run_dir)

    # Invoke scan with --resume and NO --input
    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--output",
        str(output_folder),
        "--resume",
    ])

    def fake_subprocess_run(cmd: list[str], **kwargs: object) -> MagicMock:
        script_name = cmd[1]
        if "02_scan_footage.py" in script_name:
            run_dir_idx = cmd.index("--run-dir") + 1
            run_dir_path = Path(cmd[run_dir_idx])
            (run_dir_path / "detections_refined.json").write_text('{"detections": []}')
        return MagicMock(returncode=0)

    with patch("scripts.dive_clip.subprocess.run", side_effect=fake_subprocess_run) as mock_run:
        rc = cmd_scan(args)

    assert rc == 0
    # Ingest was marked done, so only scan and render should run
    assert mock_run.call_count == 2
    # Verify the commands ran with the original run_dir
    scan_call_cmd = mock_run.call_args_list[0][0][0]
    assert str(run_dir) in scan_call_cmd
