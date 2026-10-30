"""Tests for utils/ffmpeg_streamer.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.ffmpeg_streamer import _try_create, cleanup_work_dir, extract_frames, get_work_dir


def test_get_work_dir_returns_path() -> None:
    """get_work_dir should return a Path object."""
    result = get_work_dir()
    assert isinstance(result, Path)


def test_try_create_success(tmp_path: Path) -> None:
    """_try_create should create the directory and return True."""
    new_dir = tmp_path / "new_subdir"
    assert not new_dir.exists()
    result = _try_create(new_dir)
    assert result is True
    assert new_dir.exists()


def test_try_create_already_exists(tmp_path: Path) -> None:
    """_try_create should return True if directory already exists."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    result = _try_create(existing_dir)
    assert result is True


def test_try_create_failure(tmp_path: Path) -> None:
    """_try_create should return False if creation fails."""
    # Try to create a directory inside a file (should fail)
    file_path = tmp_path / "a_file"
    file_path.write_text("not a directory")
    bad_path = file_path / "subdir"
    result = _try_create(bad_path)
    assert result is False


def test_extract_frames_success(tmp_path: Path) -> None:
    """extract_frames should parse ffmpeg output and return frame list."""
    video = tmp_path / "video.mp4"
    video.touch()
    output_dir = tmp_path / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    def create_frames(cmd, **kwargs):
        """Mock ffmpeg that creates frame files as a side effect."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_000001.jpg").write_text("fake frame 1")
        (output_dir / "frame_000002.jpg").write_text("fake frame 2")
        return mock_result

    with patch(
        "utils.ffmpeg_streamer.subprocess.run", side_effect=create_frames
    ):
        frames = extract_frames(video, output_dir, fps=1.0)

    assert len(frames) == 2
    assert frames[0]["frame_number"] == 1
    assert frames[0]["timestamp"] == 0.0  # (1-1) / 1.0 = 0
    assert frames[1]["frame_number"] == 2
    assert frames[1]["timestamp"] == 1.0  # (2-1) / 1.0 = 1
    assert "frame_path" in frames[0]


def test_extract_frames_with_start_seconds(tmp_path: Path) -> None:
    """extract_frames should offset timestamps when start_seconds is given."""
    video = tmp_path / "video.mp4"
    video.touch()
    output_dir = tmp_path / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    def create_frames(cmd, **kwargs):
        """Mock ffmpeg that creates frame files as a side effect."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_000001.jpg").write_text("fake frame")
        return mock_result

    with patch(
        "utils.ffmpeg_streamer.subprocess.run", side_effect=create_frames
    ):
        frames = extract_frames(video, output_dir, fps=1.0, start_seconds=10.0)

    assert len(frames) == 1
    assert frames[0]["timestamp"] == 10.0  # 0 + 10


def test_extract_frames_ffmpeg_failure(tmp_path: Path) -> None:
    """extract_frames should return empty list when ffmpeg fails."""
    video = tmp_path / "video.mp4"
    video.touch()
    output_dir = tmp_path / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "ffmpeg error"

    with patch(
        "utils.ffmpeg_streamer.subprocess.run", return_value=mock_result
    ):
        frames = extract_frames(video, output_dir, fps=1.0)

    assert frames == []


def test_extract_frames_timeout(tmp_path: Path) -> None:
    """extract_frames should return empty list on timeout."""
    video = tmp_path / "video.mp4"
    video.touch()
    output_dir = tmp_path / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        side_effect=subprocess.TimeoutExpired("ffmpeg", 300),
    ):
        frames = extract_frames(video, output_dir, fps=1.0)

    assert frames == []


def test_cleanup_work_dir_removes_temp_dir(tmp_path: Path) -> None:
    """cleanup_work_dir should remove a temporary directory."""
    temp_dir = tmp_path / "temp_to_clean"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("content")

    assert temp_dir.exists()
    cleanup_work_dir(temp_dir)
    assert not temp_dir.exists()


def test_cleanup_work_dir_skips_shm(tmp_path: Path) -> None:
    """cleanup_work_dir should skip /dev/shm/dive_worker_tmp."""
    shm_path = Path("/dev/shm/dive_worker_tmp")
    # Should not raise even if shm path doesn't exist
    cleanup_work_dir(shm_path)
    # Should not raise


def test_cleanup_work_dir_nonexistent(tmp_path: Path) -> None:
    """cleanup_work_dir should handle nonexistent path gracefully."""
    nonexistent = tmp_path / "does_not_exist"
    cleanup_work_dir(nonexistent)  # Should not raise
