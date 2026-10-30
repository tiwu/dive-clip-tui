"""Tests for scripts/00_ingest_media.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
