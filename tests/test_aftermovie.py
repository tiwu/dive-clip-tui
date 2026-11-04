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


def test_render_clip_with_fade_independent_flags(tmp_path: Path) -> None:
    """render_clip_with_fade should respect independent fade_in and fade_out booleans."""
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "output.mp4"

    mock_result = MagicMock(returncode=0, stderr="")

    # Case 1: fade_in=False, fade_out=True (e.g. cut in, fade out)
    with patch.object(aftermovie.subprocess, "run", return_value=mock_result) as mock_run:
        assert (
            aftermovie.render_clip_with_fade(
                source, 10.0, 15.0, output, fade_in=False, fade_out=True
            )
            is True
        )
    call_args = mock_run.call_args[0][0]
    vf_arg = next(call_args[i + 1] for i, arg in enumerate(call_args) if arg == "-vf")
    assert "fade=t=in" not in vf_arg
    assert "fade=t=out" in vf_arg

    # Case 2: fade_in=True, fade_out=False (e.g. fade in, cut out)
    with patch.object(aftermovie.subprocess, "run", return_value=mock_result) as mock_run:
        assert (
            aftermovie.render_clip_with_fade(
                source, 10.0, 15.0, output, fade_in=True, fade_out=False
            )
            is True
        )
    call_args = mock_run.call_args[0][0]
    vf_arg = next(call_args[i + 1] for i, arg in enumerate(call_args) if arg == "-vf")
    assert "fade=t=in" in vf_arg
    assert "fade=t=out" not in vf_arg

    # Case 3: fade_in=False, fade_out=False (straight hard cut)
    with patch.object(aftermovie.subprocess, "run", return_value=mock_result) as mock_run:
        assert (
            aftermovie.render_clip_with_fade(
                source, 10.0, 15.0, output, fade_in=False, fade_out=False
            )
            is True
        )
    call_args = mock_run.call_args[0][0]
    assert "-vf" not in call_args


def test_aftermovie_transition_boundary_decisions() -> None:
    """Transitions between clips should hard-cut within same clip and fade across boundaries."""
    clips = [
        {"source_file": "vidA.mp4", "start_time": 5.0, "end_time": 10.0},
        {"source_file": "vidA.mp4", "start_time": 11.0, "end_time": 15.0},  # gap 1.0s <= 3.0s
        {"source_file": "vidA.mp4", "start_time": 25.0, "end_time": 30.0},  # gap 10.0s > 3.0s
        {"source_file": "vidB.mp4", "start_time": 0.0, "end_time": 5.0},  # different file
    ]

    transitions = aftermovie.compute_clip_transitions(clips, min_transition_gap=3.0)
    assert len(transitions) == 4

    # Clip 0: first clip opens with fade_in=True; next is same file gap 1.0s -> fade_out=False
    assert transitions[0] == {"fade_in": True, "fade_out": False}
    # Clip 1: prev was same file gap 1.0s -> fade_in=False; next is same file gap 10s -> fade_out=True
    assert transitions[1] == {"fade_in": False, "fade_out": True}
    # Clip 2: prev was gap 10s -> fade_in=True; next is different file -> fade_out=True
    assert transitions[2] == {"fade_in": True, "fade_out": True}
    # Clip 3: prev was different file -> fade_in=True; last clip closes with fade_out=True
    assert transitions[3] == {"fade_in": True, "fade_out": True}


def test_aftermovie_gap_bridging() -> None:
    """Adjacent clips from the same source with small gaps should bridge into one take."""
    clips = [
        {"source_file": "vidA.mp4", "start_time": 5.0, "end_time": 8.0, "subject": "eel"},
        {"source_file": "vidA.mp4", "start_time": 8.5, "end_time": 12.0, "subject": "coral"},
        {"source_file": "vidA.mp4", "start_time": 12.5, "end_time": 22.0, "subject": "sponge"},
        {"source_file": "vidB.mp4", "start_time": 0.0, "end_time": 5.0, "subject": "turtle"},
    ]

    # bridging gap <= 1.0s with max_duration 10.0s
    bridged = aftermovie.bridge_clip_gaps(clips, bridge_gap_seconds=1.0, max_clip_duration=10.0)

    # Clips 0 and 1: gap is 0.5s, merged duration is 12.0 - 5.0 = 7.0s <= 10.0s -> merged!
    # Clip 2: gap to clip 1 is 0.5s, but merging 5.0 to 22.0 is 17.0s > 10.0s -> NOT merged!
    # Clip 3: different file -> NOT merged!
    assert len(bridged) == 3
    assert bridged[0]["start_time"] == 5.0
    assert bridged[0]["end_time"] == 12.0
    assert bridged[1]["start_time"] == 12.5
    assert bridged[1]["end_time"] == 22.0
    assert bridged[2]["source_file"] == "vidB.mp4"


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

        mock_subproc = MagicMock(returncode=0, stdout="5.0\n", stderr="")

        with (
            patch.object(aftermovie.subprocess, "run", return_value=mock_subproc),
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


def _shared_clip_id_detections(tmp_path: Path) -> dict:
    """Two highlight windows cut from the same source file (same clip_id)."""
    source = tmp_path / "source1.mp4"
    source.touch()
    return {
        "detections": [
            {
                "clip_id": "clip_152836c2cf4cd5ff",
                "file_name": "dive1.mp4",
                "file_path": str(source),
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "turtle",
                "quality_score": 9,
                "confidence": 0.9,
            },
            {
                "clip_id": "clip_152836c2cf4cd5ff",
                "file_name": "dive1.mp4",
                "file_path": str(source),
                "start_time": 30.0,
                "end_time": 34.0,
                "subject": "turtle",
                "quality_score": 8,
                "confidence": 0.8,
            },
        ]
    }


def _run_aftermovie_main(tmp_path: Path, detections: dict) -> tuple[list[Path], list[Path]]:
    """Run aftermovie main and capture rendered targets plus concatenated clip list."""
    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))

    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    rendered: list[Path] = []
    concatenated: list[Path] = []

    def mock_render(source, start, end, output, *args, **kwargs):
        rendered.append(Path(output))
        Path(output).touch()
        return True

    def mock_concat(clips, out, *args, **kwargs):
        concatenated.extend(Path(c) for c in clips)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).touch()
        return True

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

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
            "2",
        ]
        sys_module.exit = lambda code=0: None
        os.chdir(tmp_path)

        mock_subproc = MagicMock(returncode=0, stdout="5.0\n", stderr="")

        with (
            patch.object(aftermovie.subprocess, "run", return_value=mock_subproc),
            patch.object(aftermovie, "render_clip_with_fade", side_effect=mock_render),
            patch.object(aftermovie, "concat_clips", side_effect=mock_concat),
        ):
            aftermovie.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    return rendered, concatenated


def test_aftermovie_shared_clip_id_renders_distinct_temp_files(tmp_path: Path) -> None:
    """Detections sharing a clip_id must not overwrite each other's intermediate files."""
    rendered, concatenated = _run_aftermovie_main(tmp_path, _shared_clip_id_detections(tmp_path))

    assert len(rendered) == 2
    assert len(set(rendered)) == 2
    assert len(concatenated) == 2
    assert len(set(concatenated)) == 2


def test_aftermovie_cuts_fades_directly_from_source(tmp_path: Path) -> None:
    """Each highlight window should be cut and faded from the original source in one pass."""
    detections = _shared_clip_id_detections(tmp_path)

    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True, exist_ok=True)
    (highlights_dir / "clip_152836c2cf4cd5ff_10.0s-15.0s_turtle_q9.mp4").write_text("pre-rendered")

    calls: list[tuple[Path, float, float]] = []

    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))
    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    def mock_render(source, start, end, output, *args, **kwargs):
        calls.append((Path(source), float(start), float(end)))
        Path(output).touch()
        return True

    def mock_concat(clips, out, *args, **kwargs):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).touch()
        return True

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

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
            "2",
        ]
        sys_module.exit = lambda code=0: None
        os.chdir(tmp_path)

        mock_subproc = MagicMock(returncode=0, stdout="5.0\n", stderr="")

        with (
            patch.object(aftermovie.subprocess, "run", return_value=mock_subproc),
            patch.object(aftermovie, "render_clip_with_fade", side_effect=mock_render),
            patch.object(aftermovie, "concat_clips", side_effect=mock_concat),
        ):
            aftermovie.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    source_path = Path(detections["detections"][0]["file_path"])
    assert len(calls) == 2
    assert all(src == source_path for src, _, _ in calls)
    assert calls[0][1:] == (10.0, 15.0)
    assert calls[1][1:] == (30.0, 34.0)


def test_aftermovie_falls_back_to_prerendered_clip_when_source_missing(tmp_path: Path) -> None:
    """Missing source footage should fall back to the pre-rendered clip in clips_dir."""
    missing_source = tmp_path / "gone.mp4"
    detections = {
        "detections": [
            {
                "clip_id": "clip001",
                "file_name": "gone.mp4",
                "file_path": str(missing_source),
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "turtle",
                "quality_score": 9,
                "confidence": 0.9,
            }
        ]
    }

    highlights_dir = tmp_path / "outputs" / "highlights"
    highlights_dir.mkdir(parents=True)
    (highlights_dir / "clip001_10.0s-15.0s_turtle_q9.mp4").write_text("pre-rendered")

    detections_path = tmp_path / "detections_refined.json"
    detections_path.write_text(json.dumps(detections))
    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    sources: list[Path] = []

    def mock_render(source, start, end, output, *args, **kwargs):
        sources.append(Path(source))
        Path(output).touch()
        return True

    def mock_concat(clips, out, *args, **kwargs):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).touch()
        return True

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

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
        sys_module.exit = lambda code=0: None
        os.chdir(tmp_path)

        mock_subproc = MagicMock(returncode=0, stdout="5.0\n", stderr="")

        with (
            patch.object(aftermovie.subprocess, "run", return_value=mock_subproc),
            patch.object(aftermovie, "render_clip_with_fade", side_effect=mock_render),
            patch.object(aftermovie, "concat_clips", side_effect=mock_concat),
        ):
            aftermovie.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert len(sources) == 1
    assert sources[0].name == "clip001_10.0s-15.0s_turtle_q9.mp4"
