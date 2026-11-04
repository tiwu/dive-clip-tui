"""FFmpeg frame sampling utilities with RAM-disk support and error recovery.

Extracts frames from video files using FFmpeg, writing to /dev/shm
(Linux RAM-disk) when available and falling back to tempfile.mkdtemp()
on systems without shared memory (e.g. macOS).
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
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


def probe_duration(video_path: str | Path) -> float | None:
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


_probe_duration = probe_duration


def calculate_sample_timestamps(
    duration: float,
    min_count: int = 3,
    interval_seconds: float = 300.0,
    max_count: int = 8,
) -> list[float]:
    """Calculate evenly spaced timestamps across video duration.

    Args:
        duration: Video duration in seconds.
        min_count: Minimum number of sample timestamps (default: 3).
        interval_seconds: Number of seconds per additional sample frame (default: 300s / 5 min).
        max_count: Upper bound on the number of sample timestamps (default: 8).

    Returns:
        Sorted list of float timestamps.
    """
    if duration <= 0:
        return []

    min_c = max(1, min_count)
    max_c = max(min_c, max_count)
    if interval_seconds > 0:
        calculated_count = int(duration / interval_seconds)
        count = min(max_c, max(min_c, calculated_count))
    else:
        count = min_c

    step = duration / (count + 1)
    return [round((i + 1) * step, 3) for i in range(count)]


def extract_targeted_frames(
    video_path: str | Path,
    output_dir: str | Path,
    timestamps: list[float],
    scale: str | None = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Extract individual frames at specific timestamps using fast input seeking.

    Does not decode the full container, extracting only one frame per timestamp.
    Returns a list of dicts with keys: frame_path, timestamp, frame_number.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_frames: list[dict] = []
    for idx, ts in enumerate(timestamps, start=1):
        frame_file = output_dir / f"sample_{idx:03d}_{ts:.1f}s.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-ss",
            str(ts),
            "-i",
            str(video_path),
            "-vframes",
            "1",
        ]
        if scale:
            cmd.extend(["-vf", f"scale={scale}"])
        cmd.extend(["-qscale:v", "2", "-y", str(frame_file)])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout))
            if res.returncode == 0 and frame_file.exists() and frame_file.stat().st_size > 0:
                extracted_frames.append({
                    "frame_path": str(frame_file),
                    "timestamp": ts,
                    "frame_number": idx,
                })
            else:
                logger.debug(
                    "Targeted frame extraction failed at %.1fs for %s: %s",
                    ts,
                    video_path.name,
                    res.stderr.strip()[:200],
                )
        except Exception as exc:
            logger.debug(
                "Targeted frame extraction exception at %.1fs for %s: %s",
                ts,
                video_path.name,
                exc,
            )

    return extracted_frames


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


def extract_raw_frames(
    video_path: str | Path,
    fps: float = 0.5,
    size: int = 64,
    count: int | None = None,
    keyframes_only: bool = False,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    timeout: int | float | None = None,
) -> list[bytes]:
    """Extract downscaled RGB24 frames through a pipe for programmatic screening.

    The ``fps`` filter is applied the same way as :func:`extract_frames`, so frame
    ``n`` (0-based) corresponds to ``start_seconds + n / fps``. That keeps the
    prescreen metrics aligned with the JPEG frames sent to the VLM.

    ``keyframes_only`` adds ``-skip_frame nokey`` before the input so FFmpeg skips
    non-key frames during decode (Stage C fast path). Because the ``fps`` filter
    still emits one frame per interval, the output stays index-aligned with the
    regular path; if the keyframe pass yields nothing, the call retries without
    the skip flag.
    """
    video_path = Path(video_path)
    if fps <= 0 or size <= 0:
        return []

    frame_bytes = size * size * 3
    timeout_sec = int(timeout) if timeout is not None else 300

    def _run(skip_keyframes: bool) -> bytes:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if skip_keyframes:
            cmd.extend(["-skip_frame", "nokey"])
        if start_seconds is not None:
            cmd.extend(["-ss", str(start_seconds)])
        cmd.extend(["-i", str(video_path)])
        if duration_seconds is not None:
            cmd.extend(["-t", str(duration_seconds)])
        cmd.extend([
            "-vf",
            f"fps={fps},scale={size}:{size}:flags=area",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            logger.warning("Raw frame extraction timeout after %ds for %s", timeout_sec, video_path)
            return b""
        except Exception as exc:
            logger.warning("Raw frame extraction error for %s: %s", video_path, exc)
            return b""
        if result.returncode != 0:
            logger.debug(
                "Raw frame extraction returned %d for %s: %s",
                result.returncode,
                video_path,
                (result.stderr or b"").decode("utf-8", "replace").strip()[:200],
            )
            return b""
        return result.stdout or b""

    payload = _run(keyframes_only)
    if not payload and keyframes_only:
        logger.debug(
            "Keyframe-only sampling produced nothing for %s; retrying full decode", video_path
        )
        payload = _run(False)
    if not payload:
        return []

    total = len(payload) // frame_bytes
    if count is not None and count >= 0:
        total = min(total, count)
    return [payload[i * frame_bytes : (i + 1) * frame_bytes] for i in range(total)]


def compose_frame_mosaic(
    frame_paths: Sequence[str | Path],
    output_path: str | Path,
    tile: int = 2,
    padding: int = 2,
    timeout: float = 60.0,
) -> str | None:
    """Compose existing frame images into a ``tile x tile`` contact sheet.

    Frames keep their given order (left-to-right, top-to-bottom). Slots left
    empty are filled by repeating the last frame so the grid stays rectangular.
    Returns the sheet path, or ``None`` when composition fails.
    """
    paths = [str(path) for path in frame_paths]
    slot_count = max(1, tile * tile)
    if not paths or slot_count < 1:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) > slot_count:
        paths = paths[:slot_count]
    while len(paths) < slot_count:
        paths.append(paths[-1])

    pad = max(0, padding)
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
    for path in paths:
        cmd.extend(["-i", path])

    filters: list[str] = []
    for index in range(slot_count):
        if pad:
            filters.append(
                f"[{index}]pad=iw+{pad}:ih+{pad}:{pad // 2}:{pad // 2}:color=black[p{index}]"
            )
        else:
            filters.append(f"[{index}]null[p{index}]")
    layout = "|".join(f"{index % tile}*w0_{index // tile}*h0" for index in range(slot_count))
    inputs = "".join(f"[p{index}]" for index in range(slot_count))
    filters.append(f"{inputs}xstack=inputs={slot_count}:layout={layout}[out]")
    cmd.extend([
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-frames:v",
        "1",
        "-qscale:v",
        "2",
        "-y",
        str(output_path),
    ])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout))
    except Exception as exc:
        logger.debug("Frame mosaic compose failed for %s: %s", output_path, exc)
        return None
    if res.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        logger.debug(
            "Frame mosaic unusable for %s: %s",
            output_path,
            (res.stderr or "").strip()[:200],
        )
        return None
    return str(output_path)


def extract_contact_sheet(
    video_path: str | Path,
    output_dir: str | Path,
    timestamp: float,
    tile: int = 2,
    tiles: int = 4,
    interval_seconds: float = 1.0,
    scale: str = "320:-2",
    timeout: float = 60.0,
) -> dict | None:
    """Build a ``tile x tile`` temporal contact sheet around ``timestamp``.

    Tiles are ordered left-to-right, top-to-bottom and sample consecutive
    ``interval_seconds`` offsets starting at ``timestamp``. Each tile keeps its
    stable index and source timestamp so VLM results can be mapped back to the
    original timestamps. Returns ``None`` when extraction fails so the caller can
    fall back to the single-frame Pass 1 path.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    slot_count = max(1, tile * tile)
    tile_count = max(1, min(tiles, slot_count))
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles_out: list[dict] = []
    for index in range(tile_count):
        tile_ts = timestamp + index * interval_seconds
        tile_file = output_dir / f"tile_{index:02d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-ss",
            str(max(0.0, tile_ts)),
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale={scale}",
            "-qscale:v",
            "2",
            "-y",
            str(tile_file),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout))
        except Exception as exc:
            logger.debug("Contact sheet tile %d failed for %s: %s", index, video_path, exc)
            continue
        if res.returncode == 0 and tile_file.exists() and tile_file.stat().st_size > 0:
            tiles_out.append({
                "index": index,
                "timestamp": round(tile_ts, 3),
                "frame_path": str(tile_file),
            })
        else:
            logger.debug(
                "Contact sheet tile %d empty at %.2fs for %s",
                index,
                tile_ts,
                video_path,
            )

    if not tiles_out:
        return None

    sheet_path = output_dir / "contact_sheet.jpg"
    compose_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        "1",
        "-i",
        str(output_dir / "tile_%02d.jpg"),
        "-vf",
        f"tile={tile}x{tile}:padding=2",
        "-frames:v",
        "1",
        "-qscale:v",
        "2",
        "-y",
        str(sheet_path),
    ]
    try:
        res = subprocess.run(compose_cmd, capture_output=True, text=True, timeout=int(timeout))
    except Exception as exc:
        logger.debug("Contact sheet compose failed for %s: %s", video_path, exc)
        return None
    if res.returncode != 0 or not sheet_path.exists() or sheet_path.stat().st_size == 0:
        logger.debug(
            "Contact sheet compose unusable for %s: %s",
            video_path,
            res.stderr.strip()[:200],
        )
        return None

    return {
        "sheet_path": str(sheet_path),
        "tile_count": len(tiles_out),
        "tiles": tiles_out,
    }


def extract_audio_peak_levels(
    video_path: str | Path,
    timeout: float = 120.0,
) -> list[tuple[float, float]]:
    """Return ``(pts_time, peak_level_db)`` samples from FFmpeg ``astats``.

    Uses one decode pass with ``astats=metadata=1:reset=1`` and prints
    ``lavfi.astats.Overall.Peak_level`` per audio window. Clips without an audio
    track return an empty list.
    """
    video_path = Path(video_path)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-af",
        "astats=metadata=1:reset=1,ametadata=print:file=-:key=lavfi.astats.Overall.Peak_level",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(timeout))
    except Exception as exc:
        logger.debug("astats probe failed for %s: %s", video_path, exc)
        return []
    if result.returncode != 0:
        logger.debug(
            "astats probe returned %d for %s: %s",
            result.returncode,
            video_path,
            (result.stderr or "").strip()[:200],
        )
        return []
    return parse_astats_peaks(result.stdout or "")


def parse_astats_peaks(text: str) -> list[tuple[float, float]]:
    """Parse ``ametadata=print`` output into ``(pts_time, peak_db)`` tuples."""
    peaks: list[tuple[float, float]] = []
    current_time: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("frame:"):
            current_time = _parse_pts_time(stripped)
            continue
        if "Peak_level=" in stripped:
            value = _parse_peak_level(stripped)
            if value is not None:
                peaks.append((current_time if current_time is not None else 0.0, value))
    return peaks


def _parse_pts_time(line: str) -> float | None:
    """Extract ``pts_time`` from an ``ametadata`` frame header line."""
    for token in line.split():
        if token.startswith("pts_time:"):
            try:
                return float(token.split(":", 1)[1])
            except ValueError:
                return None
    return None


def _parse_peak_level(line: str) -> float | None:
    """Extract the numeric dB value from a ``Peak_level`` metadata line.

    Digital silence prints ``-inf``, which is not a usable dB measurement and
    would poison the rolling-median baseline, so non-finite values are dropped.
    """
    _, _, raw = line.partition("Peak_level=")
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None
