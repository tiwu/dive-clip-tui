"""Tests for utils/encoder.py (video encoder detection and configuration)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from utils.encoder import (
    clear_encoder_cache,
    get_fallback_encoder,
    get_video_encoder,
    probe_nvenc,
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
