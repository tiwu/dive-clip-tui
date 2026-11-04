"""Tests for utils/encoder.py (video encoder detection and configuration)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from utils.encoder import (
    VideoQuality,
    clear_encoder_cache,
    get_fallback_encoder,
    get_video_encoder,
    probe_nvenc,
    resolve_video_quality,
)


def test_probe_nvenc_success() -> None:
    clear_encoder_cache()
    mock_res = MagicMock()
    mock_res.returncode = 0

    with patch("subprocess.run", return_value=mock_res) as mock_run:
        assert probe_nvenc() is True
        assert mock_run.call_count == 1
        # Test caching: second call should not invoke subprocess
        assert probe_nvenc() is True
        assert mock_run.call_count == 1


def test_probe_nvenc_failure() -> None:
    clear_encoder_cache()
    mock_res = MagicMock()
    mock_res.returncode = 1

    with patch("subprocess.run", return_value=mock_res):
        assert probe_nvenc() is False


def test_probe_nvenc_exception() -> None:
    clear_encoder_cache()
    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg missing")):
        assert probe_nvenc() is False


def test_get_video_encoder_explicit_nvenc() -> None:
    encoder_name, args = get_video_encoder("h264_nvenc")
    assert encoder_name == "h264_nvenc"
    assert "-c:v" in args
    assert "h264_nvenc" in args
    assert "-pix_fmt" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"


def test_get_video_encoder_explicit_libx264() -> None:
    encoder_name, args = get_video_encoder("libx264", threads=2)
    assert encoder_name == "libx264"
    assert "-c:v" in args
    assert "libx264" in args
    assert "-pix_fmt" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert "-threads" in args
    assert "2" in args


def test_get_video_encoder_auto_nvenc_available() -> None:
    clear_encoder_cache()
    with patch("utils.encoder.probe_nvenc", return_value=True):
        encoder_name, args = get_video_encoder("auto")
        assert encoder_name == "h264_nvenc"
        assert "h264_nvenc" in args
        assert "-pix_fmt" in args
        assert args[args.index("-pix_fmt") + 1] == "yuv420p"


def test_get_video_encoder_auto_nvenc_unavailable() -> None:
    clear_encoder_cache()
    with patch("utils.encoder.probe_nvenc", return_value=False):
        encoder_name, args = get_video_encoder("auto", threads=4)
        assert encoder_name == "libx264"
        assert "libx264" in args
        assert "-pix_fmt" in args
        assert args[args.index("-pix_fmt") + 1] == "yuv420p"
        assert "-threads" in args
        assert "4" in args


def test_get_fallback_encoder() -> None:
    fallback = get_fallback_encoder("h264_nvenc", threads=4)
    assert fallback is not None
    name, args = fallback
    assert name == "libx264"
    assert "libx264" in args
    assert "-pix_fmt" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"

    assert get_fallback_encoder("libx264") is None


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_get_video_encoder_nvenc_uses_constant_quality_rate_control() -> None:
    """NVENC must use VBR constant-quality mode; without it ffmpeg defaults to ~2 Mbps."""
    _, args = get_video_encoder("h264_nvenc")

    assert _arg_value(args, "-preset") == "p5"
    assert _arg_value(args, "-tune") == "hq"
    assert _arg_value(args, "-rc:v") == "vbr"
    assert _arg_value(args, "-cq:v") == "19"
    assert _arg_value(args, "-b:v") == "0"
    assert _arg_value(args, "-spatial-aq") == "1"
    assert _arg_value(args, "-temporal-aq") == "1"


def test_get_video_encoder_libx264_uses_constant_rate_factor() -> None:
    """libx264 must pin a CRF value so renders do not fall back to default quality."""
    _, args = get_video_encoder("libx264", threads=2)

    assert _arg_value(args, "-crf") == "19"
    assert _arg_value(args, "-preset") == "veryfast"


def test_get_fallback_encoder_uses_constant_rate_factor() -> None:
    fallback = get_fallback_encoder("h264_nvenc", threads=4)

    assert fallback is not None
    _, args = fallback
    assert _arg_value(args, "-crf") == "19"


def test_get_video_encoder_honours_custom_quality() -> None:
    quality = VideoQuality(cq=23, crf=26)

    _, nvenc_args = get_video_encoder("h264_nvenc", quality=quality)
    _, x264_args = get_video_encoder("libx264", quality=quality)
    fallback = get_fallback_encoder("h264_nvenc", quality=quality)

    assert _arg_value(nvenc_args, "-cq:v") == "23"
    assert _arg_value(x264_args, "-crf") == "26"
    assert fallback is not None
    assert _arg_value(fallback[1], "-crf") == "26"


def test_resolve_video_quality_defaults() -> None:
    quality = resolve_video_quality(None)

    assert quality.cq == 19
    assert quality.crf == 19


def test_resolve_video_quality_reads_scan_parameters() -> None:
    config = {"scan_parameters": {"video_quality": {"cq": 21, "crf": 24}}}

    quality = resolve_video_quality(config)

    assert quality.cq == 21
    assert quality.crf == 24


def test_resolve_video_quality_ignores_invalid_values() -> None:
    config = {"scan_parameters": {"video_quality": {"cq": "high", "crf": None}}}

    quality = resolve_video_quality(config)

    assert quality.cq == 19
    assert quality.crf == 19
