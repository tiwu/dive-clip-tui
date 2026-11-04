"""Video encoder probing, selection, and hardware acceleration utilities."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_NVENC_AVAILABLE: bool | None = None

DEFAULT_CQ = 19
DEFAULT_CRF = 19


@dataclass(frozen=True)
class VideoQuality:
    """Rate-control settings applied to every encode.

    Without explicit rate control FFmpeg falls back to roughly 2 Mbps, which is
    visibly destructive for 4K source footage.
    """

    cq: int = DEFAULT_CQ
    crf: int = DEFAULT_CRF


DEFAULT_QUALITY = VideoQuality()


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def resolve_video_quality(config: Mapping[str, Any] | None = None) -> VideoQuality:
    """Build a :class:`VideoQuality` from a loaded ``config.yaml`` mapping."""
    cq = DEFAULT_CQ
    crf = DEFAULT_CRF

    if isinstance(config, Mapping):
        scan_parameters = config.get("scan_parameters")
        if isinstance(scan_parameters, Mapping):
            video_quality = scan_parameters.get("video_quality")
            if isinstance(video_quality, Mapping):
                raw_cq = _coerce_int(video_quality.get("cq"))
                if raw_cq is not None:
                    cq = raw_cq
                raw_crf = _coerce_int(video_quality.get("crf"))
                if raw_crf is not None:
                    crf = raw_crf

    return VideoQuality(cq=cq, crf=crf)


def _libx264_args(threads: int, quality: VideoQuality) -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        str(quality.crf),
        "-threads",
        str(max(1, threads)),
    ]


def _nvenc_args(quality: VideoQuality) -> list[str]:
    return [
        "-c:v",
        "h264_nvenc",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "p5",
        "-tune",
        "hq",
        "-rc:v",
        "vbr",
        "-cq:v",
        str(quality.cq),
        "-b:v",
        "0",
        "-spatial-aq",
        "1",
        "-temporal-aq",
        "1",
    ]


def clear_encoder_cache() -> None:
    """Clear cached hardware probe results (useful in tests)."""
    global _NVENC_AVAILABLE
    _NVENC_AVAILABLE = None


def probe_nvenc(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Probe whether NVIDIA NVENC (h264_nvenc) is available and functional.

    Caches the probe result so that subprocess execution occurs at most once per process.
    """
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=320x240:rate=1",
        "-c:v",
        "h264_nvenc",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "null",
        "-",
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        _NVENC_AVAILABLE = res.returncode == 0
        if _NVENC_AVAILABLE:
            logger.info("Hardware acceleration probe: NVIDIA NVENC (h264_nvenc) is available")
        else:
            logger.debug(
                "Hardware acceleration probe: NVENC unavailable (code %d: %s)",
                res.returncode,
                res.stderr.strip()[:200],
            )
    except Exception as exc:
        logger.debug("Hardware acceleration probe error: %s", exc)
        _NVENC_AVAILABLE = False

    return _NVENC_AVAILABLE


def get_video_encoder(
    configured_encoder: str = "auto",
    threads: int = 4,
    ffmpeg_bin: str = "ffmpeg",
    quality: VideoQuality | None = None,
) -> tuple[str, list[str]]:
    """Determine video encoder name and FFmpeg command arguments.

    Args:
        configured_encoder: "auto", "h264_nvenc", or "libx264".
        threads: Thread count cap for CPU encoding fallback.
        ffmpeg_bin: FFmpeg binary path or command name.
        quality: Rate-control settings; defaults to :data:`DEFAULT_QUALITY`.

    Returns:
        tuple of (encoder_name, list_of_cli_args)
    """
    normalized = (configured_encoder or "auto").strip().lower()
    effective_quality = quality or DEFAULT_QUALITY

    if normalized == "h264_nvenc" or (normalized == "auto" and probe_nvenc(ffmpeg_bin)):
        return "h264_nvenc", _nvenc_args(effective_quality)

    return "libx264", _libx264_args(threads, effective_quality)


def get_fallback_encoder(
    current_encoder: str,
    threads: int = 4,
    quality: VideoQuality | None = None,
) -> tuple[str, list[str]] | None:
    """Return fallback software encoder arguments if a hardware encoder fails."""
    if current_encoder == "h264_nvenc":
        return "libx264", _libx264_args(threads, quality or DEFAULT_QUALITY)
    return None
