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


# --- Stage C/D primitives ----------------------------------------------------


def _raw_payload(frames: list[bytes]) -> bytes:
    """Join frames into the single payload FFmpeg writes to stdout."""
    return b"".join(frames)


def _run_result(stdout: bytes = b"", returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def test_extract_raw_frames_slices_payload_into_frames() -> None:
    """The raw pipe payload is sliced on exact frame boundaries."""
    from utils.ffmpeg_streamer import extract_raw_frames

    frames = [bytes([1, 2, 3] * 4), bytes([4, 5, 6] * 4)]
    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=_run_result(_raw_payload(frames)),
    ):
        assert extract_raw_frames("clip.mp4", fps=1.0, size=2) == frames


def test_extract_raw_frames_respects_count() -> None:
    """``count`` caps the number of returned frames."""
    from utils.ffmpeg_streamer import extract_raw_frames

    frames = [bytes([1, 2, 3] * 4)] * 5
    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=_run_result(_raw_payload(frames)),
    ):
        assert len(extract_raw_frames("clip.mp4", fps=1.0, size=2, count=2)) == 2


def test_extract_raw_frames_drops_partial_trailing_frame() -> None:
    """A truncated final frame is discarded rather than returned short."""
    from utils.ffmpeg_streamer import extract_raw_frames

    payload = bytes([7] * 12) + b"\x01\x02"
    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=_run_result(payload),
    ):
        frames = extract_raw_frames("clip.mp4", fps=1.0, size=2)
    assert len(frames) == 1


def test_extract_raw_frames_rejects_invalid_sampling() -> None:
    """Non-positive fps or size short-circuits without spawning FFmpeg."""
    from utils.ffmpeg_streamer import extract_raw_frames

    with patch("utils.ffmpeg_streamer.subprocess.run") as mock_run:
        assert extract_raw_frames("clip.mp4", fps=0, size=2) == []
        assert extract_raw_frames("clip.mp4", fps=1.0, size=0) == []
    mock_run.assert_not_called()


def test_extract_raw_frames_seek_and_duration_flags() -> None:
    """``start_seconds``/``duration_seconds`` map to ffmpeg ``-ss`` and ``-t``."""
    from utils.ffmpeg_streamer import extract_raw_frames

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=_run_result(bytes([0] * 12)),
    ) as mock_run:
        extract_raw_frames("clip.mp4", fps=2.0, size=2, start_seconds=1.5, duration_seconds=4.0)
    cmd = mock_run.call_args.args[0]
    assert cmd[cmd.index("-ss") + 1] == "1.5"
    assert cmd[cmd.index("-t") + 1] == "4.0"
    assert "fps=2.0,scale=2:2:flags=area" in cmd


def test_extract_raw_frames_keyframe_only_adds_skip_frame_flag() -> None:
    """``keyframes_only`` prepends ``-skip_frame nokey`` before the input."""
    from utils.ffmpeg_streamer import extract_raw_frames

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=_run_result(bytes([0] * 12)),
    ) as mock_run:
        extract_raw_frames("clip.mp4", fps=1.0, size=2, keyframes_only=True)
    cmd = mock_run.call_args.args[0]
    assert cmd.index("-skip_frame") < cmd.index("-i")
    assert cmd[cmd.index("-skip_frame") + 1] == "nokey"


def test_extract_raw_frames_keyframe_empty_falls_back_to_full_decode() -> None:
    """An empty keyframe pass retries without ``-skip_frame``."""
    from utils.ffmpeg_streamer import extract_raw_frames

    payload = bytes([9] * 12)
    results = [_run_result(b""), _run_result(payload)]
    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=results) as mock_run:
        frames = extract_raw_frames("clip.mp4", fps=1.0, size=2, keyframes_only=True)
    assert frames == [payload]
    assert mock_run.call_count == 2
    assert "-skip_frame" not in mock_run.call_args_list[1].args[0]


def test_extract_raw_frames_missing_keyframes_and_empty_fallback_returns_empty() -> None:
    """Both passes failing yields an empty list without raising."""
    from utils.ffmpeg_streamer import extract_raw_frames

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=_run_result(b""),
    ) as mock_run:
        assert extract_raw_frames("clip.mp4", fps=1.0, size=2, keyframes_only=True) == []
    assert mock_run.call_count == 2


def test_extract_raw_frames_handles_ffmpeg_failure_and_timeout() -> None:
    """Non-zero exits and timeouts degrade to an empty frame list."""
    from utils.ffmpeg_streamer import extract_raw_frames

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=_run_result(b"", returncode=1, stderr=b"boom"),
    ):
        assert extract_raw_frames("clip.mp4", fps=1.0, size=2) == []

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1),
    ):
        assert extract_raw_frames("clip.mp4", fps=1.0, size=2) == []


def test_extract_contact_sheet_orders_tiles_and_maps_timestamps(tmp_path: Path) -> None:
    """Tiles are ordered left-to-right/top-to-bottom with per-tile timestamps."""
    from utils.ffmpeg_streamer import extract_contact_sheet

    def fake_run(cmd, **kwargs):
        target = Path(cmd[-1])
        if "tile=" not in " ".join(cmd):
            target.write_bytes(b"jpeg")
        else:
            target.write_bytes(b"sheet")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run):
        sheet = extract_contact_sheet("clip.mp4", tmp_path, timestamp=2.0, tile=2, tiles=4)

    assert sheet is not None
    assert sheet["tile_count"] == 4
    assert [tile["index"] for tile in sheet["tiles"]] == [0, 1, 2, 3]
    assert [tile["timestamp"] for tile in sheet["tiles"]] == [2.0, 3.0, 4.0, 5.0]
    assert Path(sheet["sheet_path"]).name == "contact_sheet.jpg"


def test_extract_contact_sheet_caps_tiles_to_slot_count(tmp_path: Path) -> None:
    """``tiles`` cannot exceed ``tile x tile`` slots."""
    from utils.ffmpeg_streamer import extract_contact_sheet

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"data")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run):
        sheet = extract_contact_sheet("clip.mp4", tmp_path, timestamp=0.0, tile=2, tiles=16)

    assert sheet is not None
    assert sheet["tile_count"] == 4


def test_extract_contact_sheet_keeps_partial_sheets(tmp_path: Path) -> None:
    """A sheet with some failed tiles still reports the surviving tiles."""
    from utils.ffmpeg_streamer import extract_contact_sheet

    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        is_compose = "tile=" in " ".join(cmd)
        if is_compose:
            Path(cmd[-1]).write_bytes(b"sheet")
            return MagicMock(returncode=0, stdout="", stderr="")
        if calls["n"] % 2 == 0:
            return MagicMock(returncode=1, stdout="", stderr="skip")
        Path(cmd[-1]).write_bytes(b"jpeg")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run):
        sheet = extract_contact_sheet("clip.mp4", tmp_path, timestamp=0.0, tile=2, tiles=4)

    assert sheet is not None
    assert sheet["tile_count"] == 2
    assert [tile["index"] for tile in sheet["tiles"]] == [0, 2]


def test_extract_contact_sheet_returns_none_without_tiles(tmp_path: Path) -> None:
    """Every tile failing returns ``None`` so Pass 1 falls back to single frames."""
    from utils.ffmpeg_streamer import extract_contact_sheet

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=MagicMock(returncode=1, stdout="", stderr="nope"),
    ):
        assert extract_contact_sheet("clip.mp4", tmp_path, timestamp=1.0) is None


def test_extract_contact_sheet_returns_none_when_compose_fails(tmp_path: Path) -> None:
    """A failed compose step is not reported as a usable sheet."""
    from utils.ffmpeg_streamer import extract_contact_sheet

    def fake_run(cmd, **kwargs):
        if "tile=" in " ".join(cmd):
            return MagicMock(returncode=1, stdout="", stderr="bad filter")
        Path(cmd[-1]).write_bytes(b"jpeg")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run):
        assert extract_contact_sheet("clip.mp4", tmp_path, timestamp=1.0) is None


def test_extract_contact_sheet_clamps_negative_timestamp(tmp_path: Path) -> None:
    """A negative start is clamped to zero rather than passed to FFmpeg."""
    from utils.ffmpeg_streamer import extract_contact_sheet

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"data")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run) as mock_run:
        extract_contact_sheet("clip.mp4", tmp_path, timestamp=-2.0, tile=2, tiles=1)

    first_cmd = mock_run.call_args_list[0].args[0]
    assert first_cmd[first_cmd.index("-ss") + 1] == "0.0"


def test_parse_astats_peaks_extracts_time_and_level() -> None:
    """``ametadata=print`` blocks map ``pts_time`` to ``Peak_level``."""
    from utils.ffmpeg_streamer import parse_astats_peaks

    text = (
        "frame:0    pts:0       pts_time:0\n"
        "lavfi.astats.Overall.Peak_level=-21.5\n"
        "frame:1    pts:48000   pts_time:1.5\n"
        "lavfi.astats.Overall.Peak_level=-3.25\n"
    )
    assert parse_astats_peaks(text) == [(0.0, -21.5), (1.5, -3.25)]


def test_parse_astats_peaks_skips_unparsable_lines() -> None:
    """Noise, missing timestamps, and ``-inf`` levels are tolerated."""
    from utils.ffmpeg_streamer import parse_astats_peaks

    text = (
        "\n"
        "  \n"
        "lavfi.astats.Overall.Peak_level=-inf\n"
        "frame:2    pts:0       pts_time:not-a-number\n"
        "lavfi.astats.Overall.Peak_level=-9\n"
        "unrelated output\n"
    )
    assert parse_astats_peaks(text) == [(0.0, -9.0)]


def test_parse_astats_peaks_empty_input() -> None:
    """Silence or a missing audio track yields no peaks."""
    from utils.ffmpeg_streamer import parse_astats_peaks

    assert parse_astats_peaks("") == []


def test_extract_audio_peak_levels_returns_parsed_peaks() -> None:
    """The astats command decodes audio only and returns parsed peaks."""
    from utils.ffmpeg_streamer import extract_audio_peak_levels

    stdout = "frame:0    pts:0       pts_time:0\nlavfi.astats.Overall.Peak_level=-12.0\n"
    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=MagicMock(returncode=0, stdout=stdout, stderr=""),
    ) as mock_run:
        peaks = extract_audio_peak_levels("clip.mp4")
    assert peaks == [(0.0, -12.0)]
    cmd = mock_run.call_args.args[0]
    assert "-vn" in cmd
    assert any("astats=metadata=1:reset=1" in part for part in cmd)


def test_extract_audio_peak_levels_handles_failure() -> None:
    """A failing or raising probe yields an empty peak list."""
    from utils.ffmpeg_streamer import extract_audio_peak_levels

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=MagicMock(returncode=1, stdout="", stderr="no audio"),
    ):
        assert extract_audio_peak_levels("clip.mp4") == []

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        side_effect=OSError("ffmpeg missing"),
    ):
        assert extract_audio_peak_levels("clip.mp4") == []


def test_compose_frame_mosaic_builds_xstack_command(tmp_path: Path) -> None:
    """Four frames are padded, stacked into one image, and written to disk."""
    from utils.ffmpeg_streamer import compose_frame_mosaic

    frames = [tmp_path / f"f{index}.jpg" for index in range(4)]

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"sheet")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run) as mock_run:
        out = compose_frame_mosaic(frames, tmp_path / "sheet.jpg", tile=2, padding=2)

    assert out == str(tmp_path / "sheet.jpg")
    cmd = mock_run.call_args.args[0]
    assert cmd.count("-i") == 4
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "xstack=inputs=4" in filter_complex
    assert filter_complex.count("pad=iw+2:ih+2:1:1:color=black") == 4
    assert "layout=0*w0_0*h0|1*w0_0*h0|0*w0_1*h0|1*w0_1*h0" in filter_complex


def test_compose_frame_mosaic_fills_empty_slots_with_last_frame(tmp_path: Path) -> None:
    """A short frame list is padded by repeating the final frame."""
    from utils.ffmpeg_streamer import compose_frame_mosaic

    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"sheet")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run) as mock_run:
        compose_frame_mosaic(frames, tmp_path / "sheet.jpg", tile=2, padding=0)

    cmd = mock_run.call_args.args[0]
    inputs = [cmd[i + 1] for i, part in enumerate(cmd) if part == "-i"]
    assert inputs == [str(frames[0]), str(frames[1]), str(frames[1]), str(frames[1])]
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "null" in filter_complex


def test_compose_frame_mosaic_truncates_extra_frames(tmp_path: Path) -> None:
    """More frames than slots keeps only the first ``tile x tile`` inputs."""
    from utils.ffmpeg_streamer import compose_frame_mosaic

    frames = [tmp_path / f"f{index}.jpg" for index in range(6)]

    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"sheet")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=fake_run) as mock_run:
        compose_frame_mosaic(frames, tmp_path / "sheet.jpg", tile=2)

    cmd = mock_run.call_args.args[0]
    inputs = [cmd[i + 1] for i, part in enumerate(cmd) if part == "-i"]
    assert inputs == [str(path) for path in frames[:4]]


def test_compose_frame_mosaic_returns_none_without_frames(tmp_path: Path) -> None:
    """An empty frame list short-circuits without invoking ffmpeg."""
    from utils.ffmpeg_streamer import compose_frame_mosaic

    with patch("utils.ffmpeg_streamer.subprocess.run") as mock_run:
        assert compose_frame_mosaic([], tmp_path / "sheet.jpg") is None
    mock_run.assert_not_called()


def test_compose_frame_mosaic_returns_none_on_failure(tmp_path: Path) -> None:
    """Non-zero exits, empty output, and exceptions all degrade to ``None``."""
    from utils.ffmpeg_streamer import compose_frame_mosaic

    frames = [tmp_path / "a.jpg"]

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        return_value=MagicMock(returncode=1, stdout="", stderr="boom"),
    ):
        assert compose_frame_mosaic(frames, tmp_path / "sheet.jpg", tile=2) is None

    def empty_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=empty_run):
        assert compose_frame_mosaic(frames, tmp_path / "sheet.jpg", tile=2) is None

    with patch(
        "utils.ffmpeg_streamer.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1),
    ):
        assert compose_frame_mosaic(frames, tmp_path / "sheet.jpg", tile=2) is None
