"""Tests for scripts/07_create_aftermovie.py."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys as sys_module
from pathlib import Path
from unittest.mock import MagicMock, patch

# Load scripts/07_create_aftermovie.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_aftermovie_spec = importlib.util.spec_from_file_location(
    "scripts_07_create_aftermovie", _SCRIPTS_DIR / "07_create_aftermovie.py"
)
aftermovie = importlib.util.module_from_spec(_aftermovie_spec)
_aftermovie_spec.loader.exec_module(aftermovie)


def test_aspect_ratio_filter_16_9() -> None:
    """aspect_ratio_filter('16:9') should return None (no conversion needed)."""
    assert aftermovie.aspect_ratio_filter("16:9") is None


def test_aspect_ratio_filter_9_16() -> None:
    """aspect_ratio_filter('9:16') should return a scale+pad filter string."""
    result = aftermovie.aspect_ratio_filter("9:16")
    assert result is not None
    assert "scale=1080:1920" in result
    assert "pad=1080:1920" in result


def test_aspect_ratio_filter_1_1() -> None:
    """aspect_ratio_filter('1:1') should return a scale+pad filter string."""
    result = aftermovie.aspect_ratio_filter("1:1")
    assert result is not None
    assert "scale=1080:1080" in result
    assert "pad=1080:1080" in result


def test_aspect_ratio_filter_invalid() -> None:
    """aspect_ratio_filter with invalid ratio should return None."""
    assert aftermovie.aspect_ratio_filter("invalid") is None


def test_convert_aspect_ratio_no_conversion(tmp_path: Path) -> None:
    """convert_aspect_ratio with '16:9' should just copy the file."""
    source = tmp_path / "source.mp4"
    source.write_text("fake video")
    output = tmp_path / "output.mp4"

    result = aftermovie.convert_aspect_ratio(source, output, "16:9")

    assert result is True
    assert output.exists()
    assert output.read_text() == "fake video"


def test_render_clip_with_fade_success(tmp_path: Path) -> None:
    """render_clip_with_fade should call ffmpeg with fade filters and return True."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch.object(aftermovie.subprocess, "run", return_value=mock_result) as mock_run:
        result = aftermovie.render_clip_with_fade(source, 10.0, 15.0, output)

    assert result is True
    call_args = mock_run.call_args[0][0]
    assert any("fade=t=in" in arg for arg in call_args)
    assert any("fade=t=out" in arg for arg in call_args)


def test_render_clip_with_fade_failure(tmp_path: Path) -> None:
    """render_clip_with_fade should return False when ffmpeg fails."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "ffmpeg error"

    with patch.object(aftermovie.subprocess, "run", return_value=mock_result):
        result = aftermovie.render_clip_with_fade(source, 10.0, 15.0, output)

    assert result is False


def test_add_background_music_no_music(tmp_path: Path) -> None:
    """add_background_music without music file should just copy the video."""
    video = tmp_path / "video.mp4"
    video.write_text("fake video")
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch.object(aftermovie.subprocess, "run", return_value=mock_result):
        result = aftermovie.add_background_music(video, None, output, 0.15)

    assert result is True


def test_add_background_music_with_music(tmp_path: Path) -> None:
    """add_background_music with music file should call ffmpeg with amix filter."""
    video = tmp_path / "video.mp4"
    video.write_text("fake video")
    music = tmp_path / "music.mp3"
    music.write_text("fake music")
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    with patch.object(aftermovie.subprocess, "run", return_value=mock_result) as mock_run:
        result = aftermovie.add_background_music(video, music, output, 0.15)

    assert result is True
    call_args = mock_run.call_args[0][0]
    assert any("amix" in arg for arg in call_args)
    assert any("volume=0.15" in arg for arg in call_args)


def test_add_background_music_failure(tmp_path: Path) -> None:
    """add_background_music should return False when ffmpeg fails."""
    video = tmp_path / "video.mp4"
    video.write_text("fake video")
    output = tmp_path / "output.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "mix error"

    with patch.object(aftermovie.subprocess, "run", return_value=mock_result):
        result = aftermovie.add_background_music(video, None, output, 0.15)

    assert result is False


def test_main_no_detections(tmp_path: Path) -> None:
    """main should exit gracefully when no detections found."""
    import sys as sys_module

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
            "07_create_aftermovie.py",
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
            aftermovie.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code == 0


def test_main_renders_clips(tmp_path: Path) -> None:
    """main should render clips and create aftermovie."""
    import json
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
                "quality_score": 9,
                "confidence": 0.9,
                "description": "test",
            }
        ]
    }
    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))

    # Create pre-rendered clip in outputs/highlights
    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True)
    (highlights_dir / "clip001_turtle_q9.mp4").write_text("clip 1")

    # Create source video file (fallback if no pre-rendered clip)
    (tmp_path / "source1.mp4").touch()

    # Create a minimal config.yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    # Mock subprocess.run to return success for all ffmpeg calls
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    def mock_ffmpeg(cmd, **kwargs):
        # Create output file if this is a render/concat command
        if "-i" in cmd and "-y" in cmd:
            # Find output path (last argument ending in .mp4)
            for arg in reversed(cmd):
                if arg.endswith(".mp4"):
                    Path(arg).touch()
                    break
        return mock_result

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
            "07_create_aftermovie.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "aftermovie.mp4"),
            "--config",
            str(config_path),
            "--top-n",
            "1",
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)

        with (
            patch.object(aftermovie.subprocess, "run", side_effect=mock_ffmpeg),
            contextlib.suppress(SystemExit),
        ):
            aftermovie.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code is None or exit_code == 0


def test_convert_aspect_ratio_encoder_and_fallback(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "out_9_16.mp4"

    res_fail = MagicMock(returncode=1, stderr="nvenc error")
    res_ok = MagicMock(returncode=0, stderr="")

    # Call with h264_nvenc which fails on first try, then falls back to libx264
    with patch.object(aftermovie.subprocess, "run", side_effect=[res_fail, res_ok]) as mock_run:
        ok = aftermovie.convert_aspect_ratio(source, output, "9:16", encoder_name="h264_nvenc")
        assert ok is True
        assert mock_run.call_count == 2
        # First call used h264_nvenc
        assert "h264_nvenc" in mock_run.call_args_list[0][0][0]
        # Second call used libx264 fallback
        assert "libx264" in mock_run.call_args_list[1][0][0]


def test_render_clip_with_fade_encoder_and_fallback(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "faded.mp4"

    res_fail = MagicMock(returncode=1, stderr="nvenc error")
    res_ok = MagicMock(returncode=0, stderr="")

    with patch.object(aftermovie.subprocess, "run", side_effect=[res_fail, res_ok]) as mock_run:
        ok = aftermovie.render_clip_with_fade(source, 0.0, 5.0, output, encoder_name="h264_nvenc")
        assert ok is True
        assert mock_run.call_count == 2
        assert "h264_nvenc" in mock_run.call_args_list[0][0][0]
        assert "libx264" in mock_run.call_args_list[1][0][0]


def test_aftermovie_main_forwards_selection_config(tmp_path: Path) -> None:
    """aftermovie main should forward selection.lambda_quality and selection.per_clip_soft_limit to select_diverse_highlights."""
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
        'general:\n  output_dir: "./outputs"\nselection:\n  lambda_quality: 0.72\n  per_clip_soft_limit: 3\n'
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
            "07_create_aftermovie.py",
            "--input",
            str(detections_path),
            "--output",
            str(tmp_path / "aftermovie.mp4"),
            "--config",
            str(config_path),
        ]
        sys_module.exit = lambda code=0: None
        os.chdir(tmp_path)

        def mock_concat(clips, out, *args, **kwargs):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return True

        with (
            patch.object(aftermovie, "select_diverse_highlights", side_effect=mock_select),
            patch.object(aftermovie, "render_clip_with_fade", return_value=True),
            patch.object(aftermovie, "concat_clips", side_effect=mock_concat),
        ):
            aftermovie.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert called_kwargs.get("lambda_quality") == 0.72
    assert called_kwargs.get("max_per_clip") == 3
