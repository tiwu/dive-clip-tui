"""Tests for scripts/05_render_clips.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load scripts/05_render_clips.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_render_spec = importlib.util.spec_from_file_location(
    "scripts_05_render_clips", _SCRIPTS_DIR / "05_render_clips.py"
)
render = importlib.util.module_from_spec(_render_spec)
_render_spec.loader.exec_module(render)


def test_slugify() -> None:
    """slugify should convert text to filesystem-safe slug."""
    assert render.slugify("Hello World!") == "Hello_World_"
    assert render.slugify("turtle/shark") == "turtle_shark"
    assert render.slugify("A&B*C") == "A_B_C"
    assert render.slugify("normal") == "normal"


def test_render_clip_success(tmp_path: Path) -> None:
    """render_clip should call ffmpeg with correct args and return True."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch.object(render.subprocess, "run", return_value=mock_result) as mock_run:
        result = render.render_clip(source, 10.0, 15.0, output)

    assert result is True
    # Verify ffmpeg was called with -ss and -to
    call_args = mock_run.call_args[0][0]
    assert "-ss" in call_args
    assert "-to" in call_args
    assert "10.0" in call_args or "10" in call_args
    assert "15.0" in call_args or "15" in call_args


def test_render_clip_ffmpeg_failure(tmp_path: Path) -> None:
    """render_clip should return False when ffmpeg fails."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "ffmpeg error"

    with patch.object(render.subprocess, "run", return_value=mock_result):
        result = render.render_clip(source, 10.0, 15.0, output)

    assert result is False


def test_render_clip_timeout(tmp_path: Path) -> None:
    """render_clip should handle ffmpeg timeout."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    with patch.object(
        render.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffmpeg", 300)
    ):
        result = render.render_clip(source, 10.0, 15.0, output)

    assert result is False


def test_render_clip_non_positive_duration(tmp_path: Path) -> None:
    """render_clip should return False for non-positive duration."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    with patch.object(render.subprocess, "run") as mock_run:
        result = render.render_clip(source, 10.0, 10.0, output)

    assert result is False
    mock_run.assert_not_called()


def test_render_clip_end_before_start(tmp_path: Path) -> None:
    """render_clip should return False when end_time < start_time."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    with patch.object(render.subprocess, "run") as mock_run:
        result = render.render_clip(source, 15.0, 10.0, output)

    assert result is False
    mock_run.assert_not_called()


def test_main_no_detections(tmp_path: Path, monkeypatch) -> None:
    """main should exit gracefully when no detections found."""
    import os
    import sys as sys_module

    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text('{"detections": []}')

    # Create a minimal config.yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code

    try:
        sys_module.argv = [
            "05_render_clips.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "out"),
            "--config",
            str(config_path),
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)
        render.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code == 0


def test_main_renders_clips(tmp_path: Path, monkeypatch) -> None:
    """main should render clips from detections."""
    import json
    import os
    import sys as sys_module

    # Create detections file
    detections = {
        "detections": [
            {
                "clip_id": "clip001",
                "file_name": "dive1.mp4",
                "file_path": str(tmp_path / "source1.mp4"),
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "turtle",
                "quality_score": 8,
                "confidence": 0.9,
                "description": "test",
            }
        ]
    }
    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))

    # Create source video file
    source = tmp_path / "source1.mp4"
    source.touch()

    # Create a minimal config.yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    # Mock subprocess.run to return success
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code

    try:
        sys_module.argv = [
            "05_render_clips.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "highlights"),
            "--config",
            str(config_path),
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)

        with patch.object(render.subprocess, "run", return_value=mock_result):
            render.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code is None or exit_code == 0


def test_main_logs_per_clip_progress_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Per-clip render progress must be visible without --debug."""
    import json
    import logging as logging_module
    import os
    import sys as sys_module

    caplog.set_level(logging_module.INFO)

    detections = {
        "detections": [
            {
                "clip_id": "clip001",
                "file_name": "dive1.mp4",
                "file_path": str(tmp_path / "source1.mp4"),
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "turtle",
                "quality_score": 8,
                "confidence": 0.9,
                "description": "test",
            }
        ]
    }
    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))
    (tmp_path / "source1.mp4").touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    state_path = tmp_path / ".pipeline_state.json"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

    exit_code = None

    def fake_exit(code: int = 0) -> None:
        nonlocal exit_code
        exit_code = code

    try:
        sys_module.argv = [
            "05_render_clips.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "highlights"),
            "--config",
            str(config_path),
            "--state-path",
            str(state_path),
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)
        with patch.object(render.subprocess, "run", return_value=mock_result):
            render.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code in (None, 0)
    assert "[1/1]" in caplog.text
    assert "source1.mp4" in caplog.text
    info_records = [r for r in caplog.records if "[1/1]" in r.getMessage()]
    assert info_records, "per-clip progress was not logged"
    assert all(r.levelno <= logging_module.INFO for r in info_records)

    state = json.loads(state_path.read_text())
    assert state["render_done"] is True
    assert "render_started_at" in state
    assert "render_finished_at" in state
    assert "updated_at" in state
