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
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ConfigError, load_config
from utils.encoder import (
    VideoQuality,
    get_fallback_encoder,
    get_video_encoder,
    resolve_video_quality,
)
from utils.ffmpeg_streamer import concat_clips
from utils.logging_setup import configure_logging
from utils.run_archive import resolve_and_validate_assembly_artifacts
from utils.selection import select_diverse_highlights
from utils.signals import setup_signal_handlers

logger = logging.getLogger(__name__)


def _probe_duration(video_path: Path, default: float = 5.0) -> float:
    """Return a video's duration in seconds, falling back to ``default``."""
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(probe.stdout.strip())
    except subprocess.TimeoutExpired, ValueError, FileNotFoundError, OSError:
        return default


def render_clip_with_fade(
    source_path: Path,
    start_time: float,
    end_time: float,
    output_path: Path,
    fade_duration: float = 1.0,
    fade_in: bool = True,
    fade_out: bool = True,
    encoder_name: str | None = None,
    encoder_args: list[str] | None = None,
    quality: VideoQuality | None = None,
) -> bool:
    """Render a clip with fade-in at start and/or fade-out at end.

    The fade-in/fade-out creates a cross-dissolve-like effect when clips
    are concatenated. When both fade_in and fade_out are False, the clip
    is cut straight without re-fading.
    """
    duration = end_time - start_time
    if duration <= 0:
        return False

    # Ensure fade_duration doesn't exceed clip duration
    fade = min(fade_duration, duration / 2)

    vf_filters = []
    if fade > 0:
        if fade_in:
            vf_filters.append(f"fade=t=in:st=0:d={fade}")
        if fade_out:
            vf_filters.append(f"fade=t=out:st={duration - fade}:d={fade}")

    if encoder_args is None:
        enc_name, enc_args = get_video_encoder(encoder_name or "auto", quality=quality)
    else:
        enc_name = encoder_name or "custom"
        enc_args = encoder_args

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-i",
        str(source_path),
    ]

    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    cmd.extend([
        *enc_args,
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
        fallback = get_fallback_encoder(enc_name, quality=quality)
        if fallback:
            fb_name, fb_args = fallback
            logger.warning(
                "FFmpeg aftermovie fade failed with %s for %s, falling back to %s: %s",
                enc_name,
                output_path.name,
                fb_name,
                result.stderr.strip()[:200],
            )
            return render_clip_with_fade(
                source_path,
                start_time,
                end_time,
                output_path,
                fade_duration=fade_duration,
                fade_in=fade_in,
                fade_out=fade_out,
                encoder_name=fb_name,
                encoder_args=fb_args,
                quality=quality,
            )
        logger.warning("FFmpeg failed for %s: %s", output_path.name, result.stderr.strip()[:200])
        return False

    return True


def compute_clip_transitions(
    clips: list[dict[str, Any]],
    min_transition_gap: float = 3.0,
) -> list[dict[str, bool]]:
    """Compute fade-in and fade-out decisions for each clip in an aftermovie sequence.

    Clips from different source files, or clips separated by more than min_transition_gap,
    transition via fade (dip to black). Clips continuing from the same source file separated
    by min_transition_gap or less cut directly without fading to black.
    """
    if not clips:
        return []

    transitions: list[dict[str, bool]] = []
    n = len(clips)
    for i, clip in enumerate(clips):
        curr_source = clip.get("source_file") or clip.get("file_path") or clip.get("clip_id")
        curr_start = float(clip.get("start_time", 0))
        curr_end = float(clip.get("end_time", curr_start))

        # First clip in aftermovie always fades in from black
        if i == 0:
            fade_in = True
        else:
            prev = clips[i - 1]
            prev_source = prev.get("source_file") or prev.get("file_path") or prev.get("clip_id")
            prev_end = float(prev.get("end_time", 0))
            gap = curr_start - prev_end
            fade_in = (prev_source != curr_source) or (gap > min_transition_gap)

        # Last clip in aftermovie always fades out to black
        if i == n - 1:
            fade_out = True
        else:
            next_clip = clips[i + 1]
            next_source = (
                next_clip.get("source_file")
                or next_clip.get("file_path")
                or next_clip.get("clip_id")
            )
            next_start = float(next_clip.get("start_time", 0))
            gap = next_start - curr_end
            fade_out = (curr_source != next_source) or (gap > min_transition_gap)

        transitions.append({"fade_in": fade_in, "fade_out": fade_out})

    return transitions


def bridge_clip_gaps(
    clips: list[dict[str, Any]],
    bridge_gap_seconds: float = 1.5,
    max_clip_duration: float = 12.0,
) -> list[dict[str, Any]]:
    """Bridge short temporal gaps between adjacent clips from the same source video.

    If two adjacent detections from the same source video are separated by
    bridge_gap_seconds or less, and their combined span does not exceed
    max_clip_duration, merge them into a single continuous highlight window.
    """
    if not clips:
        return []

    bridged: list[dict[str, Any]] = []
    for clip in clips:
        clip_copy = dict(clip)
        if not bridged:
            bridged.append(clip_copy)
            continue

        prev = bridged[-1]
        prev_source = prev.get("source_file") or prev.get("file_path") or prev.get("clip_id")
        curr_source = (
            clip_copy.get("source_file") or clip_copy.get("file_path") or clip_copy.get("clip_id")
        )

        curr_start = float(clip_copy.get("start_time", 0))
        curr_end = float(clip_copy.get("end_time", curr_start))
        prev_start = float(prev.get("start_time", 0))
        prev_end = float(prev.get("end_time", prev_start))

        gap = curr_start - prev_end
        merged_duration = curr_end - prev_start

        if (
            prev_source == curr_source
            and 0 <= gap <= bridge_gap_seconds
            and merged_duration <= max_clip_duration
        ):
            prev["end_time"] = curr_end
            if float(clip_copy.get("quality_score", 0)) > float(prev.get("quality_score", 0)):
                prev["quality_score"] = clip_copy["quality_score"]
                if "subject" in clip_copy:
                    prev["subject"] = clip_copy["subject"]
        else:
            bridged.append(clip_copy)

    return bridged


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
    encoder_name: str | None = None,
    quality: VideoQuality | None = None,
) -> bool:
    """Convert a video to the target aspect ratio using scale+pad."""
    vf = aspect_ratio_filter(aspect_ratio)
    if vf is None:
        # No conversion needed — just copy
        shutil.copy2(video_path, output_path)
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enc_name, enc_args = get_video_encoder(encoder_name or "auto", quality=quality)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        *enc_args,
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
        fallback = get_fallback_encoder(enc_name, quality=quality)
        if fallback:
            fb_name, fb_args = fallback
            logger.warning(
                "FFmpeg aspect ratio failed with %s for %s, falling back to %s: %s",
                enc_name,
                output_path.name,
                fb_name,
                result.stderr.strip()[:200],
            )
            retry_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                vf,
                *fb_args,
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                "-y",
                str(output_path),
            ]
            try:
                res2 = subprocess.run(retry_cmd, capture_output=True, text=True, timeout=300)
                if res2.returncode == 0:
                    return True
            except Exception:
                pass
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
        default=None,
        help=(
            "Output aftermovie file "
            "(default: <latest run>/aftermovie.mp4, else outputs/aftermovie.mp4)"
        ),
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
    parser.add_argument(
        "--min-transition-gap",
        type=float,
        default=None,
        help="Minimum gap between same-source clips before fading out/in (default: 3.0, or config)",
    )
    parser.add_argument(
        "--bridge-gap-seconds",
        type=float,
        default=None,
        help="Maximum gap between same-source clips to bridge into a single take (default: 1.5, or config)",
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
    video_quality = resolve_video_quality(config)

    # Resolve detections input path and clips directory
    try:
        detections_path, clips_dir, latest_run = resolve_and_validate_assembly_artifacts(
            output_dir,
            input_detections=args.input,
            clips_dir=args.clips_dir,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
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
    min_transition_gap = (
        args.min_transition_gap
        if args.min_transition_gap is not None
        else float(aftermovie_cfg.get("min_transition_gap", 3.0))
    )
    bridge_gap_seconds = (
        args.bridge_gap_seconds
        if args.bridge_gap_seconds is not None
        else float(aftermovie_cfg.get("bridge_gap_seconds", 1.5))
    )
    scan_params = (
        config.get("scan_parameters", {}) if isinstance(config.get("scan_parameters"), dict) else {}
    )
    max_clip_duration = float(
        aftermovie_cfg.get("max_clip_duration", scan_params.get("max_clip_duration", 12.0))
    )

    with open(detections_path, encoding="utf-8") as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        logger.info("No detections found to assemble.")
        sys.exit(0)

    # Select diverse highlights with MMR and chronological ordering
    selection_cfg = config.get("selection", {}) if isinstance(config.get("selection"), dict) else {}
    lambda_quality = float(selection_cfg.get("lambda_quality", 0.65))
    max_per_clip = (
        int(selection_cfg["per_clip_soft_limit"])
        if selection_cfg.get("per_clip_soft_limit") is not None
        else None
    )
    trace_path = (
        (latest_run / "selection_trace.json")
        if latest_run is not None
        else (output_dir / "selection_trace.json")
    )
    top_dets = select_diverse_highlights(
        detections,
        top_n=top_n,
        lambda_quality=lambda_quality,
        max_per_clip=max_per_clip,
        trace_path=trace_path,
    )

    top_dets = bridge_clip_gaps(
        top_dets,
        bridge_gap_seconds=bridge_gap_seconds,
        max_clip_duration=max_clip_duration,
    )
    transitions = compute_clip_transitions(
        top_dets,
        min_transition_gap=min_transition_gap,
    )

    logger.info(
        "Creating aftermovie from %d highlight clips (from top %d detections)",
        len(top_dets),
        min(top_n, len(detections)),
    )

    # Render each clip with smart transitions (cut within same take, fade across scenes)
    tmp_dir = output_dir / ".aftermovie_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        clip_files: list[Path] = []
        for i, det in enumerate(top_dets):
            t_flags = transitions[i]
            fade_in = t_flags["fade_in"]
            fade_out = t_flags["fade_out"]

            raw_clip_id = str(det.get("clip_id", f"clip{i}"))
            clip_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_clip_id)[:64]
            subject = det.get("subject", "unknown")
            quality_score = det.get("quality_score", 0)

            start_time = float(det.get("start_time", 0))
            end_time = float(det.get("end_time", start_time + 5))

            # A single source file can yield several highlight windows, so the
            # temp names must include the window index and its time range.
            window_tag = f"{clip_id}_{i:03d}_{start_time:.1f}s-{end_time:.1f}s"
            faded = tmp_dir / f"{window_tag}_faded.mp4"

            source_path = Path(det.get("file_path", ""))
            if source_path.is_file():
                # Preferred path: cut and fade straight from the original footage in
                # a single encode, so no generation is lost to re-encoding.
                if not render_clip_with_fade(
                    source_path,
                    start_time,
                    end_time,
                    faded,
                    fade_duration=fade_duration,
                    fade_in=fade_in,
                    fade_out=fade_out,
                    quality=video_quality,
                ):
                    logger.warning("  SKIP: failed to render %s", window_tag)
                    continue
            else:
                # Fallback: reuse a pre-rendered clip when the source is unavailable.
                matching = list(
                    clips_dir.glob(f"{clip_id}_{start_time:.1f}s-{end_time:.1f}s_*.mp4")
                )
                if not matching:
                    matching = list(clips_dir.glob(f"{clip_id}_*.mp4"))
                if not matching:
                    logger.warning("  SKIP: source not found for %s", clip_id)
                    continue

                prerendered = matching[0]
                dur = _probe_duration(prerendered)
                if not render_clip_with_fade(
                    prerendered,
                    0,
                    dur,
                    faded,
                    fade_duration=fade_duration,
                    fade_in=fade_in,
                    fade_out=fade_out,
                    quality=video_quality,
                ):
                    logger.warning("  SKIP: failed to apply fade to %s", prerendered.name)
                    continue

            clip_files.append(faded)
            logger.info(
                "  [%d/%d] %s (quality %s/10)",
                len(clip_files),
                len(top_dets),
                subject,
                quality_score,
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
        if args.output:
            output_path = Path(args.output)
        elif latest_run is not None:
            output_path = latest_run / "aftermovie.mp4"
        else:
            output_path = output_dir / "aftermovie.mp4"

        # Apply aspect ratio conversion if needed
        aspect_converted = tmp_dir / "aspect_converted.mp4"
        if args.aspect_ratio != "16:9":
            logger.info("Converting to aspect ratio %s...", args.aspect_ratio)
            if not convert_aspect_ratio(
                assembled,
                aspect_converted,
                args.aspect_ratio,
                quality=video_quality,
            ):
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
