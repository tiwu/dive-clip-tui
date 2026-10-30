"""Tests for scripts/dive_clip.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dive_clip import build_parser, cmd_scan, main, run


def test_run_success() -> None:
    """run should execute a command and return 0 on success."""
    with patch("scripts.dive_clip.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rc = run(["echo", "hello"], "test step")
    assert rc == 0


def test_run_failure() -> None:
    """run should return non-zero exit code on failure."""
    with patch("scripts.dive_clip.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        rc = run(["false"], "test step")
    assert rc == 1


def test_run_keyboard_interrupt() -> None:
    """run should handle KeyboardInterrupt and return 130."""
    with patch("scripts.dive_clip.subprocess.run", side_effect=KeyboardInterrupt()):
        rc = run(["sleep", "100"], "test step")
    assert rc == 130


def test_run_file_not_found() -> None:
    """run should return 127 when command not found."""
    with patch("scripts.dive_clip.subprocess.run", side_effect=FileNotFoundError()):
        rc = run(["nonexistent_command"], "test step")
    assert rc == 127


def test_build_parser() -> None:
    """build_parser should create a parser with scan subcommand."""
    parser = build_parser()
    assert parser is not None

    # Test scan subcommand
    args = parser.parse_args(["scan", "--input", "/tmp/footage", "--output", "/tmp/highlights"])
    assert args.command == "scan"
    assert args.input == "/tmp/footage"
    assert args.output == "/tmp/highlights"
    assert args.config == "config.yaml"
    assert args.resume is False
    assert args.reset is False
    assert args.debug is False


def test_build_parser_with_flags() -> None:
    """build_parser should parse resume, reset, and debug flags."""
    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        "/tmp/footage",
        "--output",
        "/tmp/highlights",
        "--resume",
        "--reset",
        "--debug",
    ])
    assert args.resume is True
    assert args.reset is True
    assert args.debug is True


def test_cmd_scan_skips_completed_steps(tmp_path: Path) -> None:
    """cmd_scan should skip steps that are already marked done in state."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    # Pre-create state with all steps done
    state_path = output_folder / ".pipeline_state.json"
    state_path.write_text('{"ingest_done": true, "scan_done": true, "render_done": true}')

    # Pre-create manifest and detections so the "already done" check passes
    (output_folder / "manifest_raw.json").write_text("{}")
    (output_folder / "detections_refined.json").write_text("{}")

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
        reset=False,
        debug=False,
    )

    with patch("scripts.dive_clip.subprocess.run") as mock_run:
        rc = cmd_scan(args)

    assert rc == 0
    # No subprocess calls should have been made since all steps are done
    mock_run.assert_not_called()


def test_cmd_scan_runs_all_steps(tmp_path: Path) -> None:
    """cmd_scan should run all steps when none are done."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
        reset=False,
        debug=False,
    )

    mock_result = MagicMock(returncode=0)
    with patch("scripts.dive_clip.subprocess.run", return_value=mock_result):
        rc = cmd_scan(args)

    assert rc == 0
    # Should have called subprocess 3 times (ingest, scan, render)
    assert mock_result.returncode == 0

    # State file should be created with all steps done
    state_path = output_folder / ".pipeline_state.json"
    assert state_path.exists()


def test_cmd_scan_reset_state(tmp_path: Path) -> None:
    """cmd_scan with --reset should clear state and re-run steps."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    # Pre-create state with all steps done
    state_path = output_folder / ".pipeline_state.json"
    state_path.write_text('{"ingest_done": true, "scan_done": true, "render_done": true}')

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
        reset=True,
        debug=False,
    )

    mock_result = MagicMock(returncode=0)
    with patch("scripts.dive_clip.subprocess.run", return_value=mock_result):
        rc = cmd_scan(args)

    assert rc == 0
    # Should have called subprocess 3 times since state was reset
    assert mock_result.returncode == 0


def test_main_help(capsys) -> None:
    """main should print help and exit with code 0 when --help is used."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "dive-clip" in captured.out


def test_main_scan_command(capsys, tmp_path: Path) -> None:
    """main should execute the scan command."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    mock_result = MagicMock(returncode=0)
    with patch("scripts.dive_clip.subprocess.run", return_value=mock_result):
        with pytest.raises(SystemExit) as exc_info:
            main(["scan", "--input", str(input_folder), "--output", str(output_folder)])
        assert exc_info.value.code == 0
