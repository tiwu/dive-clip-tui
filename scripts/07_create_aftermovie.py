"""Create an aftermovie with music and transitions from detection clips.

Reads detections_refined.json, sorts by quality, takes top-N clips,
renders each with fade-in/fade-out (cross-dissolve effect), concatenates
them, adds a background music track with ducking, and outputs aftermovie.mp4.
This covers Phase 2 Steps 2.2-2.4 of the simple plan.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ConfigError, load_config
from utils.logging_setup import configure_logging
from utils.run_archive import find_latest_run_dir


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
        print(f"WARNING: FFmpeg timeout rendering {output_path.name}")
        return False
    except Exception as exc:
        print(f"WARNING: FFmpeg error: {exc}")
        return False

    if result.returncode != 0:
        print(f"WARNING: FFmpeg failed for {output_path.name}: {result.stderr.strip()[:200]}")
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
        print(f"WARNING: FFmpeg timeout concatenating to {output_path.name}")
        return False
    except Exception as exc:
        print(f"WARNING: FFmpeg error: {exc}")
        return False

    list_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"WARNING: FFmpeg concat failed: {result.stderr.strip()[:200]}")
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
        import shutil

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
        print(f"WARNING: FFmpeg timeout converting aspect ratio: {output_path.name}")
        return False
    except Exception as exc:
        print(f"WARNING: FFmpeg error: {exc}")
        return False

    if result.returncode != 0:
        print(f"WARNING: FFmpeg aspect ratio conversion failed: {result.stderr.strip()[:200]}")
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
        print(f"WARNING: FFmpeg timeout adding music to {output_path.name}")
        return False
    except Exception as exc:
        print(f"WARNING: FFmpeg error: {exc}")
        return False

    if result.returncode != 0:
        print(f"WARNING: FFmpeg music mix failed: {result.stderr.strip()[:200]}")
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
        default=10,
        help="Number of top-quality clips to include (default: 10)",
    )
    parser.add_argument(
        "--music",
        default=None,
        help="Path to background music track (optional)",
    )
    parser.add_argument(
        "--music-volume",
        type=float,
        default=0.15,
        help="Background music volume relative to video audio (default: 0.15)",
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
        default=1.0,
        help="Cross-dissolve fade duration in seconds (default: 1.0)",
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
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    general = config.get("general", {})
    output_dir = Path(general.get("output_dir", "./outputs"))

    # Resolve detections input path
    detections_path = Path(args.input) if args.input else output_dir / "detections_refined.json"
    if not detections_path.exists() and not args.input:
        latest_run = find_latest_run_dir(output_dir)
        if (output_dir / "latest" / "detections_refined.json").exists():
            detections_path = output_dir / "latest" / "detections_refined.json"
        elif latest_run and (latest_run / "detections_refined.json").exists():
            detections_path = latest_run / "detections_refined.json"
    if not detections_path.exists():
        alt = Path(__file__).resolve().parent.parent / "outputs" / "detections_refined.json"
        if alt.exists():
            detections_path = alt
        else:
            print(f"ERROR: detections_refined.json not found at {detections_path}")
            print("Run 02_scan_footage.py first.")
            sys.exit(1)

    # Resolve clips directory
    clips_dir = Path(args.clips_dir) if args.clips_dir else output_dir / "highlights"
    if not clips_dir.exists() and not args.clips_dir:
        latest_run = find_latest_run_dir(output_dir)
        if (detections_path.parent / "highlights").exists():
            clips_dir = detections_path.parent / "highlights"
        elif (output_dir / "latest" / "highlights").exists():
            clips_dir = output_dir / "latest" / "highlights"
        elif latest_run and (latest_run / "highlights").exists():
            clips_dir = latest_run / "highlights"
    if not clips_dir.exists():
        print(f"ERROR: Clips directory not found: {clips_dir}")
        print("Run 05_render_clips.py first to render detection clips.")
        sys.exit(1)

    with open(detections_path) as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        print("No detections found to assemble.")
        sys.exit(0)

    # Sort by quality score (descending) and take top-N
    sorted_dets = sorted(detections, key=lambda d: d.get("quality_score", 0), reverse=True)
    top_dets = sorted_dets[: args.top_n]

    print(f"Creating aftermovie from top {len(top_dets)} of {len(detections)} detections")

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
                    print(f"  SKIP: source not found for {clip_id}")
                    continue
                if not render_clip_with_fade(
                    source_path, start_time, end_time, source, fade_duration=args.fade_duration
                ):
                    continue

            # Apply fade-in/fade-out for cross-dissolve effect
            faded = tmp_dir / f"{clip_id}_faded.mp4"
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
            )
            try:
                dur = float(duration_probe.stdout.strip())
            except ValueError:
                dur = 5.0

            if not render_clip_with_fade(source, 0, dur, faded, fade_duration=args.fade_duration):
                print(f"  SKIP: failed to apply fade to {source.name}")
                continue

            clip_files.append(faded)
            print(f"  [{len(clip_files)}/{len(top_dets)}] {subject} (quality {quality}/10)")

        if not clip_files:
            print("No clip files found to assemble.")
            sys.exit(1)

        # Concatenate all faded clips
        assembled = tmp_dir / "assembled.mp4"
        if not concat_clips(clip_files, assembled):
            print("\n✗ Assembly failed.")
            sys.exit(1)

        print(f"\n✓ Assembled {len(clip_files)} clips with cross-dissolve transitions")

        # Add background music with ducking
        music_path = Path(args.music) if args.music else None
        output_path = Path(args.output)

        # Apply aspect ratio conversion if needed
        aspect_converted = tmp_dir / "aspect_converted.mp4"
        if args.aspect_ratio != "16:9":
            print(f"\nConverting to aspect ratio {args.aspect_ratio}...")
            if not convert_aspect_ratio(assembled, aspect_converted, args.aspect_ratio):
                print("WARNING: Aspect ratio conversion failed, using original.")
                aspect_converted = assembled
        else:
            aspect_converted = assembled

        if add_background_music(
            aspect_converted, music_path, output_path, args.music_volume, args.aspect_ratio
        ):
            if music_path:
                print(f"✓ Added background music (volume {args.music_volume})")
            print(f"\n✓ Aftermovie created: {output_path}")
        else:
            print("\n✗ Failed to add music. Outputting assembled video without music.")
            import shutil

            shutil.copy2(assembled, output_path)
            print(f"\n✓ Aftermovie created (no music): {output_path}")

    except KeyboardInterrupt:
        print("\nAftermovie creation interrupted by user (Ctrl+C). Cleaning up...")
        sys.exit(130)
    finally:
        # Clean up temp files
        import shutil as _shutil

        _shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
