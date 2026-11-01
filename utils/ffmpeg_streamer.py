"""FFmpeg frame sampling utilities with RAM-disk support and error recovery.

Extracts frames from video files using FFmpeg, writing to /dev/shm
(Linux RAM-disk) when available and falling back to tempfile.mkdtemp()
on systems without shared memory (e.g. macOS).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def get_work_dir(use_ramdisk: bool = True) -> Path:
    """Return a private temp directory in /dev/shm if requested and available, else standard temp.

    When `use_ramdisk` is True and `/dev/shm` is available and writable (e.g. Linux),
    temporary frames are written to shared memory to maximize I/O throughput and eliminate
    SSD write wear. When False or on systems without `/dev/shm` (e.g. macOS), falls back
    to standard temporary directory.
    """
    shm_base = Path("/dev/shm")
    target_dir = (
        shm_base if use_ramdisk and shm_base.is_dir() and os.access(shm_base, os.W_OK) else None
    )
    return Path(tempfile.mkdtemp(prefix="dive_worker_", dir=target_dir))


def _try_create(path: Path) -> bool:
    """Try to create the shm directory. Returns True on success."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def _probe_duration(video_path: Path) -> float | None:
    """Probe video duration in seconds using ffprobe, or None if probe fails."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass
    return None


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 0.5,
    scale: str | None = None,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    timeout: int | float | None = None,
) -> list[dict]:
    """Extract frames from a video file using FFmpeg.

    Returns a list of dicts with keys: frame_path, timestamp, frame_number.
    Skips corrupt frames and logs warnings instead of crashing the batch.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    # Clean any stale frames from previous runs in this output dir
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute dynamic timeout based on duration if not explicitly passed
    if timeout is None:
        dur = duration_seconds
        if dur is None:
            dur = _probe_duration(video_path)
        timeout = max(300, int(dur * 2.0)) if dur is not None and dur > 0 else 300
    timeout_sec = int(timeout)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]

    if start_seconds is not None:
        cmd.extend(["-ss", str(start_seconds)])
    cmd.extend(["-i", str(video_path)])

    if duration_seconds is not None:
        cmd.extend(["-t", str(duration_seconds)])

    vf_filters = [f"fps={fps}"]
    if scale:
        vf_filters.append(f"scale={scale}")
    cmd.extend(["-vf", ",".join(vf_filters)])

    output_pattern = str(output_dir / "frame_%06d.jpg")
    cmd.extend(["-qscale:v", "2", output_pattern])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timeout after %ds for %s", timeout_sec, video_path)
        return []
    except Exception as exc:
        logger.warning("FFmpeg error for %s: %s", video_path, exc)
        return []

    if result.returncode != 0:
        logger.warning(
            "FFmpeg returned %d for %s: %s",
            result.returncode,
            video_path,
            result.stderr.strip(),
        )

    frames = []
    for frame_file in sorted(output_dir.glob("frame_*.jpg")):
        try:
            frame_num = int(frame_file.stem.split("_")[1])
        except ValueError, IndexError:
            continue
        timestamp = (frame_num - 1) / fps if fps > 0 else 0.0
        if start_seconds is not None:
            timestamp += start_seconds
        frames.append({
            "frame_path": str(frame_file),
            "timestamp": timestamp,
            "frame_number": frame_num,
        })

    # Fallback for short clips where the fps filter produced 0 frames
    if not frames and video_path.exists() and video_path.stat().st_size > 0:
        fallback_pattern = str(output_dir / "frame_000001.jpg")
        fallback_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
        if start_seconds is not None:
            fallback_cmd.extend(["-ss", str(start_seconds)])
        fallback_cmd.extend(["-i", str(video_path), "-vframes", "1"])
        if scale:
            fallback_cmd.extend(["-vf", f"scale={scale}"])
        fallback_cmd.extend(["-qscale:v", "2", fallback_pattern])
        try:
            subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)
            fallback_file = Path(fallback_pattern)
            if fallback_file.exists() and fallback_file.stat().st_size > 0:
                frames.append({
                    "frame_path": str(fallback_file),
                    "timestamp": start_seconds or 0.0,
                    "frame_number": 1,
                })
        except Exception as exc:
            logger.warning("Fallback frame extraction failed for %s: %s", video_path, exc)

    return frames


def cleanup_work_dir(path: str | Path) -> None:
    """Clean up temporary frame files."""
    path = Path(path)
    if not path.exists():
        return
    if path in (Path("/dev/shm"), Path("/dev/shm/dive_worker_tmp"), Path("/"), Path("/tmp")):
        return
    shutil.rmtree(path, ignore_errors=True)


def concat_clips(clip_paths: list[Path], output_path: Path) -> bool:
    """Concatenate MP4 clips using FFmpeg's concat demuxer.

    All clips must have compatible codec/resolution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg concat demuxer requires absolute or relative paths in the list file.
    list_path = output_path.parent / "concat_list.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as f:
            for clip in clip_paths:
                path_str = str(clip.resolve())
                if "\n" in path_str or "\r" in path_str:
                    raise ValueError(f"Path contains newline character: {path_str}")
                # FFmpeg concat demuxer escaping for single-quoted string: escape \ then '
                escaped = path_str.replace("\\", "\\\\").replace("'", "\\'")
                f.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg timeout concatenating to %s", output_path.name)
            return False
        except Exception as exc:
            logger.warning("FFmpeg error: %s", exc)
            return False

        if result.returncode != 0:
            logger.warning("FFmpeg concat failed: %s", (result.stderr or "").strip()[:200])
            return False

        return True
    finally:
        list_path.unlink(missing_ok=True)
