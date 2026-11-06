"""Folder ingestion: walk a directory, enumerate videos, extract metadata, emit JSON manifest.

This is Phase 1 Step 1.1 of the simple plan. Points at a folder of videos
and produces a plain JSON manifest with duration, FPS, resolution, and codec.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging_setup import configure_logging
from utils.signals import setup_signal_handlers
from utils.telemetry import classify_telemetry_type, detect_telemetry_stream

logger = logging.getLogger(__name__)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg"}

HIGH_FPS_THRESHOLD = 100.0


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    import re

    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:40]


def _parse_rate(rate: str) -> float:
    """Parse a rational ffprobe frame rate string (``"30000/1001"``) into a float."""
    try:
        num, den = rate.split("/")
        return round(float(num) / float(den), 2) if float(den) > 0 else 0.0
    except ValueError, ZeroDivisionError:
        return 0.0


def _iter_boxes(handle: BinaryIO, start: int, end: int):
    """Yield ``(box_type, payload_offset, payload_size)`` for MP4 boxes in ``[start, end)``."""
    offset = start
    while offset + 8 <= end:
        handle.seek(offset)
        header = handle.read(8)
        if len(header) < 8:
            return
        size, box_type = struct.unpack(">I4s", header)
        header_len = 8
        if size == 1:
            large = handle.read(8)
            if len(large) < 8:
                return
            size = struct.unpack(">Q", large)[0]
            header_len = 16
        elif size == 0:
            size = end - offset
        if size < header_len:
            return
        payload_size = min(size - header_len, end - offset - header_len)
        yield box_type.decode("latin-1"), offset + header_len, payload_size
        offset += size


def parse_hilight_tags(data: bytes) -> list[float]:
    """Parse a raw ``hmmt`` HiLight Marker Table payload into sorted second offsets."""
    if len(data) < 8:
        return []
    # 4 bytes version/flags, 4 bytes entry count, then ``count`` * 8-byte entries
    # of ``(uint32 time_ms, uint32 reserved)``.
    count = struct.unpack(">I", data[4:8])[0]
    available = (len(data) - 8) // 8
    count = min(count, available)
    tags: list[float] = []
    for index in range(count):
        entry_offset = 8 + index * 8
        time_ms = struct.unpack(">I", data[entry_offset : entry_offset + 4])[0]
        if time_ms == 0xFFFFFFFF:  # no-value marker written by some firmwares
            continue
        tags.append(round(time_ms / 1000.0, 3))
    return sorted(set(tags))


def read_hilight_tags(video_path: Path, max_moov_bytes: int = 64 * 1024 * 1024) -> list[float]:
    """Read manual GoPro HiLight markers from ``moov.udta.hmmt`` (empty on any failure)."""
    try:
        with open(video_path, "rb") as handle:
            file_size = video_path.stat().st_size
            for box_type, payload_offset, payload_size in _iter_boxes(handle, 0, file_size):
                if box_type not in ("moov", "udta") or payload_size > max_moov_bytes:
                    continue
                handle.seek(payload_offset)
                container = handle.read(payload_size)
                for tags in _find_hmmt_tags(container):
                    return tags
    except OSError as exc:
        logger.debug("Could not read HiLight tags from %s: %s", video_path, exc)
    return []


def _find_hmmt_tags(container: bytes) -> list[list[float]]:
    """Recursively scan an in-memory MP4 box container for ``hmmt`` payloads."""
    found: list[list[float]] = []
    for box_type, payload_offset, payload_size in _iter_boxes(
        io.BytesIO(container), 0, len(container)
    ):
        if payload_size <= 0 or payload_offset + payload_size > len(container):
            continue
        payload = container[payload_offset : payload_offset + payload_size]
        if box_type == "hmmt":
            tags = parse_hilight_tags(payload)
            if tags:
                found.append(tags)
        elif box_type in ("udta", "meta", "ilst"):
            # ``meta`` uses a 4-byte version/flags prefix before child boxes.
            if box_type == "meta" and len(payload) > 4:
                payload = payload[4:]
            found.extend(_find_hmmt_tags(payload))
    return found


def _bits_per_pixel_density(width: int, height: int, fps: float, bit_rate: float) -> float:
    """Return bits per pixel per frame (compression-density proxy), 0.0 when unknown."""
    pixels = float(width) * float(height) * float(fps)
    if pixels <= 0 or bit_rate <= 0:
        return 0.0
    return round(bit_rate / pixels, 4)


def probe_video(video_path: Path) -> dict:
    """Extract container metadata for all streams from a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,codec_tag_string,width,height,r_frame_rate,pix_fmt",
        "-show_entries",
        "format=duration,bit_rate:format_tags=creation_time",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {
                "file_name": video_path.name,
                "file_path": str(video_path),
                "error": result.stderr.strip() or "ffprobe failed",
            }

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if stream is None:
            stream = streams[0] if streams else {}

        duration = float(fmt.get("duration", 0)) if fmt.get("duration") else 0.0
        fps = _parse_rate(str(stream.get("r_frame_rate", "0/1")))
        width = stream.get("width") or 0
        height = stream.get("height") or 0
        try:
            bit_rate = float(fmt.get("bit_rate") or 0.0)
        except TypeError, ValueError:
            bit_rate = 0.0

        format_tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}
        creation_time = str(format_tags.get("creation_time", "") or "")

        stream_tags = {str(t).lower() for t in (s.get("codec_tag_string") or "" for s in streams)}
        detected_tel = detect_telemetry_stream(video_path, streams=streams)
        telemetry_type = detected_tel[0] if detected_tel else classify_telemetry_type(stream_tags)
        telemetry_stream_index = detected_tel[1] if detected_tel else None
        hilight_tags = read_hilight_tags(video_path)

        return {
            "file_name": video_path.name,
            "file_path": str(video_path),
            "duration": round(duration, 2),
            "fps": fps,
            "resolution": f"{width or '?'}x{height or '?'}",
            "codec": stream.get("codec_name", "unknown"),
            "pix_fmt": stream.get("pix_fmt", "unknown"),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "has_telemetry": telemetry_type != "",
            "telemetry_type": telemetry_type,
            "telemetry_stream_index": telemetry_stream_index,
            "is_high_fps": fps >= HIGH_FPS_THRESHOLD,
            "creation_time": creation_time,
            "bit_rate": bit_rate,
            "bitrate_density": _bits_per_pixel_density(width, height, fps, bit_rate),
            "hilight_tags": hilight_tags,
        }

    except subprocess.TimeoutExpired:
        return {
            "file_name": video_path.name,
            "file_path": str(video_path),
            "error": "ffprobe timeout",
        }
    except Exception as exc:
        return {
            "file_name": video_path.name,
            "file_path": str(video_path),
            "error": str(exc),
        }


def ingest_folder(folder: Path) -> list[dict]:
    """Walk a folder and collect metadata for all video files."""
    video_paths = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    clips = []
    for index, video_path in enumerate(video_paths, start=1):
        logger.debug("Probing %d/%d: %s", index, len(video_paths), video_path.name)
        meta = probe_video(video_path)
        if "error" in meta:
            logger.warning("Skipped %s: %s", video_path.name, meta["error"])
        clips.append(meta)
    return clips


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ingest a folder of videos and emit a JSON manifest"
    )
    parser.add_argument("--input", required=True, help="Path to footage folder")
    parser.add_argument(
        "--output", default="outputs/manifest.json", help="Output manifest JSON path"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--trace", action="store_true", help="Enable trace logging")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append this run's log lines to the given file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    configure_logging(debug=args.debug, trace=args.trace, log_file=args.log_file)
    setup_signal_handlers()

    folder = Path(args.input)
    if not folder.exists():
        logger.error("Folder does not exist: %s", folder)
        sys.exit(1)

    logger.info("Scanning %s for video files", folder)
    try:
        clips = ingest_folder(folder)
    except KeyboardInterrupt:
        logger.warning("\nIngest interrupted by user (Ctrl+C).")
        sys.exit(130)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project_name": folder.name,
        "total_clips": len(clips),
        "clips": clips,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Manifest written: %s (%d clips)", output_path, len(clips))


if __name__ == "__main__":
    main()
