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


def test_render_detections_renders_new_and_skips_existing(tmp_path: Path) -> None:
    """render_detections should render new clips and skip existing clips with non-zero size."""
    source_video = tmp_path / "source.mp4"
    source_video.touch()
    highlights_dir = tmp_path / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create the first highlight clip to simulate existing file
    existing_clip = highlights_dir / "c1_0.0s-5.0s_turtle_q8.mp4"
    existing_clip.write_bytes(b"existing video content")

    detections = [
        {
            "clip_id": "c1",
            "file_path": str(source_video),
            "start_time": 0.0,
            "end_time": 5.0,
            "subject": "turtle",
            "quality_score": 8,
        },
        {
            "clip_id": "c2",
            "file_path": str(source_video),
            "start_time": 10.0,
            "end_time": 15.0,
            "subject": "shark",
            "quality_score": 9,
        },
    ]

    rendered_files = []

    def mock_render_clip(source, start, end, out):
        rendered_files.append(out.name)
        out.write_bytes(b"newly rendered")
        return True

    with patch.object(render, "render_clip", side_effect=mock_render_clip):
        rendered_count, failed_count = render.render_detections(detections, highlights_dir)

    assert rendered_count == 1
    assert failed_count == 0
    # c1_0.0s-5.0s_turtle_q8.mp4 was skipped, only c2_10.0s-15.0s_shark_q9.mp4 was rendered
    assert rendered_files == ["c2_10.0s-15.0s_shark_q9.mp4"]
    # Existing clip content was untouched
    assert existing_clip.read_bytes() == b"existing video content"


def test_render_detections_multiple_detections_same_subject_unique_filenames(
    tmp_path: Path,
) -> None:
    """Multiple detections of the same subject and quality in the same clip should render to distinct files."""
    source_video = tmp_path / "source.mp4"
    source_video.touch()
    highlights_dir = tmp_path / "highlights"

    detections = [
        {
            "clip_id": "c1",
            "file_path": str(source_video),
            "start_time": 0.0,
            "end_time": 5.0,
            "subject": "turtle",
            "quality_score": 8,
        },
        {
            "clip_id": "c1",
            "file_path": str(source_video),
            "start_time": 10.0,
            "end_time": 15.0,
            "subject": "turtle",
            "quality_score": 8,
        },
    ]

    rendered_files = []

    def mock_render_clip(source, start, end, out):
        rendered_files.append(out.name)
        out.write_bytes(b"content")
        return True

    with patch.object(render, "render_clip", side_effect=mock_render_clip):
        rendered_count, failed_count = render.render_detections(detections, highlights_dir)

    assert rendered_count == 2
    assert failed_count == 0
    assert len(rendered_files) == 2
    assert rendered_files[0] == "c1_0.0s-5.0s_turtle_q8.mp4"
    assert rendered_files[1] == "c1_10.0s-15.0s_turtle_q8.mp4"


def test_render_detections_quiet_mode(tmp_path: Path) -> None:
    """render_detections(log_to_console=False) should pass extra={'no_console': True}."""
    source_video = tmp_path / "source.mp4"
    source_video.touch()
    highlights_dir = tmp_path / "highlights"

    detections = [
        {
            "clip_id": "c1",
            "file_path": str(source_video),
            "start_time": 0.0,
            "end_time": 5.0,
            "subject": "turtle",
            "quality_score": 8,
        }
    ]

    with (
        patch.object(render, "render_clip", return_value=True),
        patch.object(render.logger, "info") as mock_info,
    ):
        rendered, failed = render.render_detections(
            detections, highlights_dir, log_to_console=False
        )

    assert rendered == 1
    assert failed == 0
    assert mock_info.called
    assert mock_info.call_args[1].get("extra", {}).get("no_console") is True
