"""Tests for scripts/06_assemble_video.py."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys as sys_module
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load scripts/06_assemble_video.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_assemble_spec = importlib.util.spec_from_file_location(
    "scripts_06_assemble_video", _SCRIPTS_DIR / "06_assemble_video.py"
)
assemble = importlib.util.module_from_spec(_assemble_spec)
_assemble_spec.loader.exec_module(assemble)


def test_concat_clips_success(tmp_path: Path) -> None:
    """concat_clips should call ffmpeg with concat demuxer and return True."""
    clip1 = tmp_path / "clip1.mp4"
    clip2 = tmp_path / "clip2.mp4"
    clip1.touch()
    clip2.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch("utils.ffmpeg_streamer.subprocess.run", return_value=mock_result) as mock_run:
        result = assemble.concat_clips([clip1, clip2], output)

    assert result is True
    call_args = mock_run.call_args[0][0]
    assert "-f" in call_args
    assert "concat" in call_args
    assert "-safe" in call_args
    assert "0" in call_args


def test_concat_clips_failure(tmp_path: Path) -> None:
    """concat_clips should return False when ffmpeg fails."""
    clip1 = tmp_path / "clip1.mp4"
    clip1.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "concat error"

    with patch("utils.ffmpeg_streamer.subprocess.run", return_value=mock_result):
        result = assemble.concat_clips([clip1], output)

    assert result is False


def test_concat_clips_timeout(tmp_path: Path) -> None:
    """concat_clips should handle ffmpeg timeout."""
    clip1 = tmp_path / "clip1.mp4"
    clip1.touch()
    output = tmp_path / "output.mp4"

    with patch(
        "utils.ffmpeg_streamer.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 300)
    ):
        result = assemble.concat_clips([clip1], output)

    assert result is False


def test_main_no_detections(tmp_path: Path) -> None:
    """main should exit gracefully when no detections found."""

    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text('{"detections": []}')

    # Create outputs/highlights dir (default clips dir)
    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

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
            "06_assemble_video.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "aftermovie.mp4"),
            "--config",
            str(config_path),
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)
        with contextlib.suppress(SystemExit):
            assemble.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code == 0


def test_main_renders_clips(tmp_path: Path) -> None:
    """main should render top-N clips from detections."""
    import os

    # Create detections file with 3 detections
    detections = {
        "detections": [
            {
                "clip_id": "clip001",
                "file_name": "dive1.mp4",
                "file_path": str(tmp_path / "source1.mp4"),
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "turtle",
                "quality_score": 9,
                "confidence": 0.9,
                "description": "test",
            },
            {
                "clip_id": "clip002",
                "file_name": "dive2.mp4",
                "file_path": str(tmp_path / "source2.mp4"),
                "start_time": 20.0,
                "end_time": 25.0,
                "subject": "shark",
                "quality_score": 8,
                "confidence": 0.8,
                "description": "test",
            },
        ]
    }
    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))

    # Create rendered clips in outputs/highlights (default clips dir)
    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True)
    (highlights_dir / "clip001_turtle_q9.mp4").write_text("clip 1")
    (highlights_dir / "clip002_shark_q8.mp4").write_text("clip 2")

    # Create source video files
    (tmp_path / "source1.mp4").touch()
    (tmp_path / "source2.mp4").touch()

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
        raise SystemExit(code)

    try:
        sys_module.argv = [
            "06_assemble_video.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "aftermovie.mp4"),
            "--config",
            str(config_path),
            "--top-n",
            "2",
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)

        with (
            patch("utils.ffmpeg_streamer.subprocess.run", return_value=mock_result),
            contextlib.suppress(SystemExit),
        ):
            assemble.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code is None or exit_code == 0


def test_concat_clips_escapes_single_quotes(tmp_path: Path) -> None:
    """concat_clips should escape single quotes using FFmpeg demuxer syntax."""
    clip1 = tmp_path / "it's_a_test.mp4"
    clip1.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 0

    written_content: list[str] = []

    def mock_run(cmd, **kwargs):
        list_file = Path(cmd[cmd.index("-i") + 1])
        written_content.append(list_file.read_text(encoding="utf-8"))
        return mock_result

    with patch("utils.ffmpeg_streamer.subprocess.run", side_effect=mock_run):
        assert assemble.concat_clips([clip1], output) is True

    assert len(written_content) == 1
    assert "it\\'s_a_test.mp4" in written_content[0]


def test_concat_clips_rejects_newlines(tmp_path: Path) -> None:
    """concat_clips should raise ValueError if path contains a newline."""
    output = tmp_path / "output.mp4"
    malicious_path = MagicMock(spec=Path)
    malicious_path.resolve.return_value = "clip.mp4\nfile 'malicious.mp4'"

    with pytest.raises(ValueError, match="Path contains newline character"):
        assemble.concat_clips([malicious_path], output)


def test_main_forwards_selection_config(tmp_path: Path) -> None:
    """main should forward selection.lambda_quality and selection.per_clip_soft_limit to select_diverse_highlights."""
    detections = {
        "detections": [
            {
                "clip_id": "clip001",
                "file_name": "dive1.mp4",
                "file_path": str(tmp_path / "source1.mp4"),
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "turtle",
                "quality_score": 9,
                "confidence": 0.9,
            },
        ]
    }
    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))

    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True)
    (highlights_dir / "clip001_turtle_q9.mp4").write_text("clip 1")
    (tmp_path / "source1.mp4").touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'general:\n  output_dir: "./outputs"\nselection:\n  lambda_quality: 0.85\n  per_clip_soft_limit: 2\n'
    )

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

    called_kwargs: dict = {}

    def mock_select(dets, **kwargs):
        nonlocal called_kwargs
        called_kwargs = kwargs
        return dets

    try:
        sys_module.argv = [
            "06_assemble_video.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "out.mp4"),
            "--config",
            str(config_path),
        ]
        sys_module.exit = lambda code=0: None
        os.chdir(tmp_path)
        with (
            patch.object(assemble, "select_diverse_highlights", side_effect=mock_select),
            patch.object(assemble, "concat_clips", return_value=True),
        ):
            assemble.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert called_kwargs.get("lambda_quality") == 0.85
    assert called_kwargs.get("max_per_clip") == 2


def test_assemble_mixes_atmosphere_music_and_skips_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """06_assemble_video.py applies music mixing for --atmosphere ocean_ambient and skips for none."""
    import urllib.error

    monkeypatch.setattr(
        "utils.audio_music.urllib.request.urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    detections = {
        "detections": [
            {
                "clip_id": "clip001",
                "file_name": "dive1.mp4",
                "file_path": str(tmp_path / "source1.mp4"),
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "turtle",
                "quality_score": 9,
                "confidence": 0.9,
            },
        ]
    }
    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections), encoding="utf-8")

    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True)
    (highlights_dir / "clip001_turtle_q9.mp4").write_text("clip 1")
    (tmp_path / "source1.mp4").touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'general:\n  output_dir: "./outputs"\nmusic:\n  atmosphere: "ocean_ambient"\n'
    )

    def fake_concat(clips, out):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"concat_mp4")
        return True

    mix_calls: list[tuple[Path, Path | None, float, float | None, bool]] = []

    def fake_mix(video, music, out, **kwargs):
        mix_calls.append((
            video,
            music,
            kwargs.get("music_volume"),
            kwargs.get("original_volume"),
            kwargs.get("scuba_filter"),
        ))
        Path(out).write_bytes(b"mixed_mp4")
        return True

    original_argv = sys_module.argv
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        # Case 1: --atmosphere ocean_ambient -> mixes music + 35% original audio
        sys_module.argv = [
            "06_assemble_video.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "reel_ambient.mp4"),
            "--config",
            str(config_path),
            "--atmosphere",
            "ocean_ambient",
        ]
        with (
            patch.object(assemble, "concat_clips", side_effect=fake_concat),
            patch.object(assemble, "probe_duration", return_value=5.0),
            patch.object(assemble, "mix_video_audio", side_effect=fake_mix),
        ):
            assemble.main()

        assert len(mix_calls) == 1
        assert mix_calls[0][1] is not None
        assert mix_calls[0][1].name == "ocean_ambient.wav"
        assert mix_calls[0][2] == 0.65
        assert mix_calls[0][3] == 0.35
        assert mix_calls[0][4] is True
        assert (tmp_path / "reel_ambient.mp4").read_bytes() == b"mixed_mp4"

        # Case 2: --atmosphere none -> skips mixing
        mix_calls.clear()
        sys_module.argv = [
            "06_assemble_video.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "reel_none.mp4"),
            "--config",
            str(config_path),
            "--atmosphere",
            "none",
        ]
        with (
            patch.object(assemble, "concat_clips", side_effect=fake_concat),
            patch.object(assemble, "probe_duration", return_value=5.0),
            patch.object(assemble, "mix_video_audio", side_effect=fake_mix),
        ):
            assemble.main()

        assert mix_calls == []
        assert (tmp_path / "reel_none.mp4").read_bytes() == b"concat_mp4"
    finally:
        sys_module.argv = original_argv
        os.chdir(original_cwd)
