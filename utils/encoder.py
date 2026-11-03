"""Video encoder probing, selection, and hardware acceleration utilities."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_NVENC_AVAILABLE: bool | None = None


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
) -> tuple[str, list[str]]:
    """Determine video encoder name and FFmpeg command arguments.

    Args:
        configured_encoder: "auto", "h264_nvenc", or "libx264".
        threads: Thread count cap for CPU encoding fallback.
        ffmpeg_bin: FFmpeg binary path or command name.

    Returns:
        tuple of (encoder_name, list_of_cli_args)
    """
    normalized = (configured_encoder or "auto").strip().lower()

    if normalized == "h264_nvenc" or (normalized == "auto" and probe_nvenc(ffmpeg_bin)):
        return "h264_nvenc", ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4"]

    return "libx264", [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-threads",
        str(max(1, threads)),
    ]


def get_fallback_encoder(
    current_encoder: str,
    threads: int = 4,
) -> tuple[str, list[str]] | None:
    """Return fallback software encoder arguments if a hardware encoder fails."""
    if current_encoder == "h264_nvenc":
        return "libx264", [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-threads",
            str(max(1, threads)),
        ]
    return None
