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


def test_get_work_dir_uses_shm_when_enabled_and_available(tmp_path: Path) -> None:
    """get_work_dir should use /dev/shm when use_ramdisk=True and /dev/shm is available."""
    shm_fake = tmp_path / "dev_shm"
    shm_fake.mkdir()

    with (
        patch("utils.ffmpeg_streamer.Path") as mock_path,
        patch("tempfile.mkdtemp", return_value=str(shm_fake / "dive_worker_123")) as mock_mkdtemp,
        patch("os.access", return_value=True),
    ):
        mock_shm = MagicMock()
        mock_shm.is_dir.return_value = True
        mock_path.side_effect = lambda p: mock_shm if p == "/dev/shm" else Path(p)

        res = get_work_dir(use_ramdisk=True)
        assert res == Path(shm_fake / "dive_worker_123")
        mock_mkdtemp.assert_called_once_with(prefix="dive_worker_", dir=mock_shm)


def test_get_work_dir_ignores_shm_when_use_ramdisk_false(tmp_path: Path) -> None:
    """get_work_dir should use standard temp dir when use_ramdisk=False even if /dev/shm exists."""
    fallback_dir = tmp_path / "temp_fallback"
    fallback_dir.mkdir()

    with patch("tempfile.mkdtemp", return_value=str(fallback_dir)) as mock_mkdtemp:
        res = get_work_dir(use_ramdisk=False)
        assert res == fallback_dir
        mock_mkdtemp.assert_called_once_with(prefix="dive_worker_", dir=None)


def test_get_work_dir_falls_back_when_shm_missing(tmp_path: Path) -> None:
    """get_work_dir should fall back to standard temp when /dev/shm is missing or not writable."""
    fallback_dir = tmp_path / "temp_fallback"
    fallback_dir.mkdir()

    with (
        patch("utils.ffmpeg_streamer.Path") as mock_path,
        patch("tempfile.mkdtemp", return_value=str(fallback_dir)) as mock_mkdtemp,
    ):
        mock_shm = MagicMock()
        mock_shm.is_dir.return_value = False
        mock_path.side_effect = lambda p: mock_shm if p == "/dev/shm" else Path(p)

        res = get_work_dir(use_ramdisk=True)
        assert res == fallback_dir
        mock_mkdtemp.assert_called_once_with(prefix="dive_worker_", dir=None)


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

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=create_frames):
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

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=create_frames):
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

    with patch("utils.ffmpeg_streamer.subprocess.run", return_value=mock_result):
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


def test_extract_frames_dynamic_timeout(tmp_path: Path) -> None:
    """extract_frames should scale timeout dynamically with video duration."""
    video = tmp_path / "video.mp4"
    video.touch()
    output_dir = tmp_path / "frames"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("utils.ffmpeg_streamer.subprocess.run", return_value=mock_result) as mock_run:
        extract_frames(video, output_dir, duration_seconds=600.0)

    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["timeout"] >= 1200


def test_extract_frames_short_clip_fallback(tmp_path: Path) -> None:
    """extract_frames should fall back to extracting 1 frame at start when fps filter yields 0 frames."""
    video = tmp_path / "short_video.mp4"
    video.write_bytes(b"dummy video data")
    output_dir = tmp_path / "frames"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    call_count = 0

    def mock_subprocess(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        if "-vframes" in cmd:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "frame_000001.jpg").write_text("fallback frame")
        return mock_result

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=mock_subprocess):
        frames = extract_frames(video, output_dir, fps=0.5)

    assert len(frames) == 1
    assert frames[0]["frame_number"] == 1
    assert frames[0]["timestamp"] == 0.0
    assert call_count == 3  # probe duration, ffmpeg fps pass, fallback ffmpeg frame


def test_concat_clips_success(tmp_path: Path) -> None:
    """concat_clips should invoke ffmpeg concat demuxer and succeed."""
    from utils.ffmpeg_streamer import concat_clips

    c1 = tmp_path / "c1.mp4"
    c2 = tmp_path / "c2.mp4"
    c1.touch()
    c2.touch()
    out = tmp_path / "out.mp4"

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stderr = ""

    with patch("utils.ffmpeg_streamer.subprocess.run", return_value=mock_res) as mock_run:
        assert concat_clips([c1, c2], out) is True

    call_args = mock_run.call_args[0][0]
    assert "-f" in call_args
    assert "concat" in call_args
    assert not (tmp_path / "concat_list.txt").exists()


def test_concat_clips_failure_and_cleanup(tmp_path: Path) -> None:
    """concat_clips should clean up list file and return False on failure."""
    from utils.ffmpeg_streamer import concat_clips

    c1 = tmp_path / "c1.mp4"
    c1.touch()
    out = tmp_path / "out.mp4"

    mock_res = MagicMock()
    mock_res.returncode = 1
    mock_res.stderr = "fail"

    with patch("utils.ffmpeg_streamer.subprocess.run", return_value=mock_res):
        assert concat_clips([c1], out) is False
    assert not (tmp_path / "concat_list.txt").exists()


def test_concat_clips_rejects_newlines(tmp_path: Path) -> None:
    """concat_clips should raise ValueError on newline in path."""
    from utils.ffmpeg_streamer import concat_clips

    bad_path = tmp_path / "clip\n.mp4"
    out = tmp_path / "out.mp4"

    with pytest.raises(ValueError, match="Path contains newline character"):
        concat_clips([bad_path], out)
