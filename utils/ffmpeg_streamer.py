"""FFmpeg frame sampling utilities with RAM-disk support and error recovery.

Extracts frames from video files using FFmpeg, writing to /dev/shm
(Linux RAM-disk) when available and falling back to tempfile.mkdtemp()
on systems without shared memory (e.g. macOS).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


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


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 0.5,
    scale: str | None = None,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"WARNING: FFmpeg timeout for {video_path}")
        return []
    except Exception as exc:
        print(f"WARNING: FFmpeg error for {video_path}: {exc}")
        return []

    if result.returncode != 0:
        print(
            f"WARNING: FFmpeg returned {result.returncode} for {video_path}: "
            f"{result.stderr.strip()}"
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

    return frames


def cleanup_work_dir(path: str | Path) -> None:
    """Clean up temporary frame files."""
    path = Path(path)
    if not path.exists():
        return
    if path in (Path("/dev/shm"), Path("/dev/shm/dive_worker_tmp"), Path("/"), Path("/tmp")):
        return
    shutil.rmtree(path, ignore_errors=True)
