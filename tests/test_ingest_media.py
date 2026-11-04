"""Tests for scripts/00_ingest_media.py."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load scripts/00_ingest_media.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_ingest_spec = importlib.util.spec_from_file_location(
    "scripts_00_ingest_media", _SCRIPTS_DIR / "00_ingest_media.py"
)
ingest = importlib.util.module_from_spec(_ingest_spec)
_ingest_spec.loader.exec_module(ingest)


def test_slugify() -> None:
    """slugify should convert text to filesystem-safe slug."""
    assert ingest.slugify("Hello World!") == "Hello_World_"
    assert ingest.slugify("turtle/shark") == "turtle_shark"
    assert ingest.slugify("A&B*C") == "A_B_C"
    assert ingest.slugify("normal") == "normal"


def test_probe_video_success(tmp_path: Path) -> None:
    """probe_video should parse ffprobe JSON output correctly."""
    fake_video = tmp_path / "dive.mp4"
    fake_video.touch()

    ffprobe_output = {
        "streams": [
            {
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "25/1",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
            }
        ],
        "format": {"duration": "60.5"},
    }

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(ffprobe_output)
    mock_result.stderr = ""

    with patch.object(ingest.subprocess, "run", return_value=mock_result):
        result = ingest.probe_video(fake_video)

    assert result["file_name"] == "dive.mp4"
    assert result["file_path"] == str(fake_video)
    assert result["duration"] == 60.5
    assert result["fps"] == 25.0
    assert result["resolution"] == "1920x1080"
    assert result["codec"] == "h264"
    assert result["pix_fmt"] == "yuv420p"
    assert "error" not in result


def test_probe_video_ffprobe_failure(tmp_path: Path) -> None:
    """probe_video should return error dict when ffprobe fails."""
    fake_video = tmp_path / "dive.mp4"
    fake_video.touch()

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "No such file"

    with patch.object(ingest.subprocess, "run", return_value=mock_result):
        result = ingest.probe_video(fake_video)

    assert result["error"] == "No such file"


def test_probe_video_timeout(tmp_path: Path) -> None:
    """probe_video should handle ffprobe timeout."""
    fake_video = tmp_path / "dive.mp4"
    fake_video.touch()

    import subprocess

    with patch.object(
        ingest.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffprobe", 60)
    ):
        result = ingest.probe_video(fake_video)

    assert result["error"] == "ffprobe timeout"


def test_ingest_folder_no_videos(tmp_path: Path) -> None:
    """ingest_folder with no video files should return empty list."""
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    with patch.object(ingest, "probe_video") as mock_probe:
        result = ingest.ingest_folder(empty_folder)

    assert result == []
    mock_probe.assert_not_called()


def test_ingest_folder_with_videos(tmp_path: Path) -> None:
    """ingest_folder should walk the folder and probe each video file."""
    video_folder = tmp_path / "videos"
    video_folder.mkdir()
    (video_folder / "dive1.mp4").touch()
    (video_folder / "dive2.mov").touch()
    (video_folder / "._dive1.mp4").touch()
    (video_folder / "not_video.txt").touch()  # should be ignored

    def fake_probe(path: Path) -> dict:
        return {"file_name": path.name, "file_path": str(path), "duration": 10.0}

    with patch.object(ingest, "probe_video", side_effect=fake_probe):
        result = ingest.ingest_folder(video_folder)

    assert len(result) == 2
    names = {r["file_name"] for r in result}
    assert names == {"dive1.mp4", "dive2.mov"}


def test_probe_video_invalid_fps(tmp_path: Path) -> None:
    """probe_video should handle invalid frame rate gracefully."""
    fake_video = tmp_path / "dive.mp4"
    fake_video.touch()

    ffprobe_output = {
        "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "0/0", "codec_name": "h264"}],
        "format": {"duration": "10.0"},
    }

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(ffprobe_output)
    mock_result.stderr = ""

    with patch.object(ingest.subprocess, "run", return_value=mock_result):
        result = ingest.probe_video(fake_video)

    # 0/0 should cause ZeroDivisionError, caught and fps set to 0.0
    assert result["fps"] == 0.0


def test_main_success(tmp_path: Path, monkeypatch) -> None:
    """main should ingest folder and write manifest."""
    import os
    import sys as sys_module

    # Create a fake video file
    video_folder = tmp_path / "footage"
    video_folder.mkdir()
    (video_folder / "dive1.mp4").touch()

    output_path = tmp_path / "outputs" / "manifest_raw.json"

    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({
        "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "25/1", "codec_name": "h264"}],
        "format": {"duration": "60.0"},
    })
    mock_result.stderr = ""

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    try:
        sys_module.argv = [
            "00_ingest_media.py",
            "--input",
            str(video_folder),
            "--output",
            str(output_path),
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)

        with patch.object(ingest.subprocess, "run", return_value=mock_result):
            ingest.main()
    except SystemExit:
        pass
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code is None or exit_code == 0
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data["total_clips"] == 1
    assert data["clips"][0]["file_name"] == "dive1.mp4"


def test_main_accepts_debug_flag(tmp_path: Path) -> None:
    """main should accept --debug so dive-clip can thread the flag through."""
    csv_path = tmp_path / "manifest.json"
    parser = ingest.build_parser()
    args = parser.parse_args([
        "--input",
        str(tmp_path),
        "--output",
        str(csv_path),
        "--debug",
    ])
    assert args.debug is True

    default_args = parser.parse_args(["--input", str(tmp_path), "--output", str(csv_path)])
    assert default_args.debug is False


def test_main_debug_flag_runs_end_to_end(tmp_path: Path) -> None:
    """--debug must not break the ingest run."""
    import os

    (tmp_path / "footage").mkdir()
    fake_video = tmp_path / "footage" / "dive1.mp4"
    fake_video.touch()
    output_path = tmp_path / "manifest_raw.json"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({
        "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "25/1", "codec_name": "h264"}],
        "format": {"duration": "10.0"},
    })
    mock_result.stderr = ""

    original_argv = sys.argv
    sys_module = sys
    original_exit = sys_module.exit
    original_cwd = Path.cwd()
    exit_code = None

    def fake_exit(code: int = 0) -> None:
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    try:
        sys_module.argv = [
            "00_ingest_media.py",
            "--input",
            str(tmp_path / "footage"),
            "--output",
            str(output_path),
            "--debug",
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)

        with patch.object(ingest.subprocess, "run", return_value=mock_result):
            ingest.main()
    except SystemExit:
        pass
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code is None or exit_code == 0
    assert output_path.exists()


# --- Stage A: container telemetry, HiLight markers, probe fields ------------


def _box(box_type: str, payload: bytes) -> bytes:
    """Build a single MP4 box with a 32-bit size."""
    return struct.pack(">I4s", len(payload) + 8, box_type.encode("latin-1")) + payload


def _hmmt_payload(marker_ms: list[int], count: int | None = None) -> bytes:
    """Build an ``hmmt`` payload: version/flags, entry count, then 8-byte entries."""
    entries = b"".join(struct.pack(">II", ms, 0) for ms in marker_ms)
    declared = len(marker_ms) if count is None else count
    return struct.pack(">II", 0, declared) + entries


def _probe_result(streams: list[dict], fmt: dict) -> MagicMock:
    """Build a fake successful ffprobe ``subprocess.run`` result."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"streams": streams, "format": fmt})
    mock_result.stderr = ""
    return mock_result


def test_probe_video_reports_audio_and_telemetry_streams(tmp_path: Path) -> None:
    """probe_video should classify every stream, not just the first video one."""
    fake_video = tmp_path / "dive.mp4"
    fake_video.touch()

    mock_result = _probe_result(
        [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "hevc",
                "codec_tag_string": "hvc1",
                "width": 3840,
                "height": 2160,
                "r_frame_rate": "60/1",
                "pix_fmt": "yuv420p10le",
            },
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "codec_tag_string": "mp4a"},
            {
                "index": 2,
                "codec_type": "data",
                "codec_name": "bin_data",
                "codec_tag_string": "gpmd",
            },
        ],
        {
            "duration": "45.25",
            "bit_rate": "100000000",
            "tags": {"creation_time": "2024-05-01T10:00:00.000000Z"},
        },
    )

    with patch.object(ingest.subprocess, "run", return_value=mock_result):
        result = ingest.probe_video(fake_video)

    assert result["resolution"] == "3840x2160"
    assert result["pix_fmt"] == "yuv420p10le"
    assert result["has_audio"] is True
    assert result["has_telemetry"] is True
    assert result["telemetry_type"] == "gopro"
    assert result["is_high_fps"] is False
    assert result["creation_time"] == "2024-05-01T10:00:00.000000Z"
    assert result["bit_rate"] == 100_000_000.0
    # 100 Mbit/s over 3840x2160x60 pixels
    assert result["bitrate_density"] == pytest.approx(0.2009, abs=1e-4)


def test_probe_video_without_extra_streams(tmp_path: Path) -> None:
    """Plain H.264 clips report no audio, no telemetry, and no HiLight markers."""
    fake_video = tmp_path / "plain.mp4"
    fake_video.touch()

    mock_result = _probe_result(
        [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "codec_tag_string": "avc1",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "25/1",
            }
        ],
        {"duration": "10.0"},
    )

    with patch.object(ingest.subprocess, "run", return_value=mock_result):
        result = ingest.probe_video(fake_video)

    assert result["has_audio"] is False
    assert result["has_telemetry"] is False
    assert result["telemetry_type"] == ""
    assert result["hilight_tags"] == []
    assert result["creation_time"] == ""
    assert result["bit_rate"] == 0.0
    assert result["bitrate_density"] == 0.0


@pytest.mark.parametrize(
    ("frame_rate", "expected"),
    [("24/1", False), ("99/1", False), ("100/1", True), ("120/1", True), ("240/1", True)],
)
def test_probe_video_flags_high_fps(tmp_path: Path, frame_rate: str, expected: bool) -> None:
    """Clips at or above 100 fps should be flagged as high frame rate."""
    fake_video = tmp_path / "clip.mp4"
    fake_video.touch()

    mock_result = _probe_result(
        [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": frame_rate,
            }
        ],
        {"duration": "5.0"},
    )

    with patch.object(ingest.subprocess, "run", return_value=mock_result):
        result = ingest.probe_video(fake_video)

    assert result["is_high_fps"] is expected


def test_parse_hilight_tags_reads_millisecond_entries() -> None:
    """parse_hilight_tags should convert millisecond entries to sorted seconds."""
    tags = ingest.parse_hilight_tags(_hmmt_payload([12500, 4000, 12500, 900]))

    assert tags == [0.9, 4.0, 12.5]


def test_parse_hilight_tags_ignores_placeholder_entries() -> None:
    """Sentinel ``0xFFFFFFFF`` entries written by some firmwares are skipped."""
    payload = (
        struct.pack(">II", 0, 2) + struct.pack(">II", 0xFFFFFFFF, 0) + struct.pack(">II", 2000, 0)
    )

    assert ingest.parse_hilight_tags(payload) == [2.0]


def test_parse_hilight_tags_truncated_payload_is_ignored() -> None:
    """A count larger than the payload must not read past the available bytes."""
    payload = struct.pack(">II", 0, 99) + struct.pack(">II", 3000, 0)

    assert ingest.parse_hilight_tags(payload) == [3.0]
    assert ingest.parse_hilight_tags(b"") == []


def test_read_hilight_tags_walks_moov_udta(tmp_path: Path) -> None:
    """read_hilight_tags should find ``moov/udta/hmmt`` in a real MP4 layout."""
    video = tmp_path / "hilight.mp4"
    hmmt = _box("hmmt", _hmmt_payload([7000, 1500]))
    udta = _box("udta", _box("meta", b"\x00\x00\x00\x00" + hmmt))
    video.write_bytes(_box("ftyp", b"isom") + _box("moov", udta) + _box("mdat", b"\x00" * 16))

    assert ingest.read_hilight_tags(video) == [1.5, 7.0]


def test_read_hilight_tags_without_markers_returns_empty(tmp_path: Path) -> None:
    """Clips without an ``hmmt`` box (or unreadable files) yield no markers."""
    video = tmp_path / "nomarkers.mp4"
    video.write_bytes(_box("ftyp", b"isom") + _box("moov", _box("udta", b"")))

    assert ingest.read_hilight_tags(video) == []
    assert ingest.read_hilight_tags(tmp_path / "missing.mp4") == []


def test_probe_video_includes_hilight_tags(tmp_path: Path) -> None:
    """probe_video should surface HiLight markers found in the container."""
    video = tmp_path / "marked.mp4"
    video.write_bytes(
        _box("ftyp", b"isom")
        + _box("moov", _box("udta", _box("hmmt", _hmmt_payload([4200]))))
        + _box("mdat", b"\x00" * 8)
    )

    mock_result = _probe_result(
        [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
            }
        ],
        {"duration": "12.0"},
    )

    with patch.object(ingest.subprocess, "run", return_value=mock_result):
        result = ingest.probe_video(video)

    assert result["hilight_tags"] == [4.2]
