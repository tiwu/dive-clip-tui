"""Create an aftermovie with music and transitions from detection clips.

Reads detections_refined.json, sorts by quality, takes top-N clips,
renders each with fade-in/fade-out (cross-dissolve effect), concatenates
them, adds a background music track with ducking, and outputs aftermovie.mp4.
This covers Phase 2 Steps 2.2-2.4 of the simple plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ConfigError, load_config
from utils.logging_setup import configure_logging
from utils.run_archive import resolve_assembly_artifacts

logger = logging.getLogger(__name__)


def setup_signal_handlers() -> None:
    """Ensure SIGINT and SIGTERM raise KeyboardInterrupt for clean shutdown."""

    def _handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except ValueError, AttributeError:
        pass


def render_clip_with_fade(
    source_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    fade_duration: float = 1.0,
) -> bool:
    """Render a clip with fade-in at start and fade-out at end.

    The fade-in/fade-out creates a cross-dissolve-like effect when clips
    are concatenated.
    """
    duration = end_time - start_time
    if duration <= 0:
        return False

    # Ensure fade_duration doesn't exceed clip duration
    fade = min(fade_duration, duration / 2)

    vf_filters = []
    if fade > 0:
        vf_filters.append(f"fade=t=in:st=0:d={fade}")
        vf_filters.append(f"fade=t=out:st={duration - fade}:d={fade}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
    ]

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    cmd.extend([
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timeout rendering %s", output_path.name)
        return False
    except Exception as exc:
        logger.warning("FFmpeg error: %s", exc)
        return False

    if result.returncode != 0:
        logger.warning("FFmpeg failed for %s: %s", output_path.name, result.stderr.strip()[:200])
        return False

    return True


def concat_clips(clip_paths: list[Path], output_path: Path) -> bool:
    """Concatenate MP4 clips using FFmpeg's concat demuxer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    list_path = output_path.parent / "concat_list.txt"
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

    list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.warning("FFmpeg concat failed: %s", result.stderr.strip()[:200])
        return False

    return True


def aspect_ratio_filter(aspect_ratio: str) -> str | None:
    """Return an FFmpeg vf filter string for the target aspect ratio.

    Uses scale+pad with force_original_aspect_ratio=decrease to preserve
    all content (letterbox/pillarbox) rather than cropping.
    """
    if aspect_ratio == "16:9":
        return None
    if aspect_ratio == "9:16":
        return (
            "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
        )
    if aspect_ratio == "1:1":
        return (
            "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2"
        )
    return None


def convert_aspect_ratio(
    video_path: Path,
    output_path: Path,
    aspect_ratio: str,
) -> bool:
    """Convert a video to the target aspect ratio using scale+pad."""
    vf = aspect_ratio_filter(aspect_ratio)
    if vf is None:
        # No conversion needed — just copy
        shutil.copy2(video_path, output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timeout converting aspect ratio: %s", output_path.name)
        return False
    except Exception as exc:
        logger.warning("FFmpeg error: %s", exc)
        return False

    if result.returncode != 0:
        logger.warning("FFmpeg aspect ratio conversion failed: %s", result.stderr.strip()[:200])
        return False

    return True


def add_background_music(
    video_path: Path,
    music_path: Path | None,
    output_path: Path,
    music_volume: float = 0.15,
    aspect_ratio: str = "16:9",
) -> bool:
    """Mix a background music track under the video audio (basic ducking).

    The music is mixed at a reduced volume so it sits under the video's
    original audio rather than overpowering it. Optionally converts the
    video to the target aspect ratio first.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if music_path is None or not music_path.exists():
        # No music — just copy the video
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-y",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(music_path),
            "-filter_complex",
            (
                f"[1:a]aloop=loop=-1:size=2^{16},volume={music_volume}[music];"
                f"[0:a][music]amix=inputs=2:duration=first[aout]"
            ),
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(output_path),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timeout adding music to %s", output_path.name)
        return False
    except Exception as exc:
        logger.warning("FFmpeg error: %s", exc)
        return False

    if result.returncode != 0:
        logger.warning("FFmpeg music mix failed: %s", result.stderr.strip()[:200])
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an aftermovie with music and transitions from detection clips"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to detections_refined.json (default: outputs/detections_refined.json)",
    )
    parser.add_argument(
        "--clips-dir",
        default=None,
        help="Folder with rendered clips (default: outputs/highlights)",
    )
    parser.add_argument(
        "--output",
        default="outputs/aftermovie.mp4",
        help="Output aftermovie file (default: outputs/aftermovie.mp4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top-quality clips to include (default: 10, or config)",
    )
    parser.add_argument(
        "--music",
        default=None,
        help="Path to background music track (optional)",
    )
    parser.add_argument(
        "--music-volume",
        type=float,
        default=None,
        help="Background music volume relative to video audio (default: 0.15, or config)",
    )
    parser.add_argument(
        "--aspect-ratio",
        choices=["16:9", "9:16", "1:1"],
        default="16:9",
        help="Target aspect ratio for output video (default: 16:9)",
    )
    parser.add_argument(
        "--fade-duration",
        type=float,
        default=None,
        help="Cross-dissolve fade duration in seconds (default: 1.0, or config)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional logfile path",
    )
    args = parser.parse_args()
    setup_signal_handlers()
    configure_logging(debug=args.debug, log_file=args.log_file)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    general = config.get("general", {})
    output_dir = Path(general.get("output_dir", "./outputs"))

    # Resolve detections input path and clips directory
    detections_path, clips_dir, _ = resolve_assembly_artifacts(
        output_dir,
        input_detections=args.input,
        clips_dir=args.clips_dir,
    )
    if not detections_path.exists():
        alt = Path(__file__).resolve().parent.parent / "outputs" / "detections_refined.json"
        if alt.exists():
            detections_path = alt
        else:
            logger.error(
                "detections_refined.json not found at %s. Run 02_scan_footage.py first.",
                detections_path,
            )
            sys.exit(1)

    if not clips_dir.exists():
        logger.error(
            "Clips directory not found: %s. Run 05_render_clips.py first to render detection clips.",
            clips_dir,
        )
        sys.exit(1)

    aftermovie_cfg = (
        config.get("aftermovie", {}) if isinstance(config.get("aftermovie"), dict) else {}
    )
    top_n = args.top_n if args.top_n is not None else int(aftermovie_cfg.get("top_n_clips", 10))
    music_volume = (
        args.music_volume
        if args.music_volume is not None
        else float(aftermovie_cfg.get("music_volume", 0.15))
    )
    fade_duration = (
        args.fade_duration
        if args.fade_duration is not None
        else float(aftermovie_cfg.get("fade_duration", 1.0))
    )

    with open(detections_path, encoding="utf-8") as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        logger.info("No detections found to assemble.")
        sys.exit(0)

    # Sort by quality score (descending) and take top-N
    sorted_dets = sorted(detections, key=lambda d: d.get("quality_score", 0), reverse=True)
    top_dets = sorted_dets[:top_n]

    logger.info("Creating aftermovie from top %d of %d detections", len(top_dets), len(detections))

    # Render each clip with fade-in/fade-out (cross-dissolve effect)
    tmp_dir = output_dir / ".aftermovie_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        clip_files: list[Path] = []
        for i, det in enumerate(top_dets):
            raw_clip_id = str(det.get("clip_id", f"clip{i}"))
            clip_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_clip_id)[:64]
            subject = det.get("subject", "unknown")
            quality = det.get("quality_score", 0)

            start_time = float(det.get("start_time", 0))
            end_time = float(det.get("end_time", start_time + 5))

            # Look for pre-rendered clip (timestamped first, then any clip_id match)
            matching = list(clips_dir.glob(f"{clip_id}_{start_time:.1f}s-{end_time:.1f}s_*.mp4"))
            if not matching:
                matching = list(clips_dir.glob(f"{clip_id}_*.mp4"))
            if matching:
                source = matching[0]
            else:
                # Render from source video
                source_path = Path(det.get("file_path", ""))
                source = tmp_dir / f"{clip_id}_rendered.mp4"
                if not source_path.exists():
                    logger.warning("  SKIP: source not found for %s", clip_id)
                    continue
                if not render_clip_with_fade(
                    source_path, start_time, end_time, source, fade_duration=fade_duration
                ):
                    continue

            # Apply fade-in/fade-out for cross-dissolve effect
            faded = tmp_dir / f"{clip_id}_faded.mp4"
            try:
                duration_probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1",
                        str(source),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                dur = float(duration_probe.stdout.strip())
            except subprocess.TimeoutExpired, ValueError:
                dur = 5.0

            if not render_clip_with_fade(source, 0, dur, faded, fade_duration=fade_duration):
                logger.warning("  SKIP: failed to apply fade to %s", source.name)
                continue

            clip_files.append(faded)
            logger.info(
                "  [%d/%d] %s (quality %s/10)", len(clip_files), len(top_dets), subject, quality
            )

        if not clip_files:
            logger.error("No clip files found to assemble.")
            sys.exit(1)

        # Concatenate all faded clips
        assembled = tmp_dir / "assembled.mp4"
        if not concat_clips(clip_files, assembled):
            logger.error("✗ Assembly failed.")
            sys.exit(1)

        logger.info("✓ Assembled %d clips with cross-dissolve transitions", len(clip_files))

        # Add background music with ducking
        music_path = Path(args.music) if args.music else None
        output_path = Path(args.output)

        # Apply aspect ratio conversion if needed
        aspect_converted = tmp_dir / "aspect_converted.mp4"
        if args.aspect_ratio != "16:9":
            logger.info("Converting to aspect ratio %s...", args.aspect_ratio)
            if not convert_aspect_ratio(assembled, aspect_converted, args.aspect_ratio):
                logger.warning("Aspect ratio conversion failed, using original.")
                aspect_converted = assembled
        else:
            aspect_converted = assembled

        if add_background_music(
            aspect_converted, music_path, output_path, music_volume, args.aspect_ratio
        ):
            if music_path:
                logger.info("✓ Added background music (volume %s)", music_volume)
            logger.info("✓ Aftermovie created: %s", output_path)
        else:
            logger.warning("Failed to add music. Outputting assembled video without music.")
            shutil.copy2(assembled, output_path)
            logger.info("✓ Aftermovie created (no music): %s", output_path)

    except KeyboardInterrupt:
        logger.warning("Aftermovie creation interrupted by user (Ctrl+C). Cleaning up...")
        sys.exit(130)
    finally:
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
