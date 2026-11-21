"""Assemble top-N detection clips into a single highlight video.

Reads detections_refined.json, sorts by quality score, takes the top-N,
and concatenates the rendered MP4 clips using FFmpeg's concat demuxer.
This is Phase 2 Step 2.1 of the simple plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.audio_music import (
    VALID_ATMOSPHERE_CHOICES,
    mix_video_audio,
    resolve_atmosphere_music,
    resolve_music_config,
)
from utils.config import ConfigError, load_config
from utils.ffmpeg_streamer import concat_clips, probe_duration
from utils.logging_setup import configure_logging
from utils.run_archive import resolve_and_validate_assembly_artifacts
from utils.selection import select_diverse_highlights
from utils.signals import setup_signal_handlers

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble top-N detection clips into a single highlight video"
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
        help="Output video file (default: outputs/aftermovie.mp4)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top-quality clips to include (default: 10, or config)",
    )
    parser.add_argument(
        "--atmosphere",
        choices=VALID_ATMOSPHERE_CHOICES,
        default=None,
        help="Copyright-free music atmosphere preset (default: from config, or ocean_ambient)",
    )
    parser.add_argument(
        "--music",
        default=None,
        help="Path to background music track (optional; overrides --atmosphere)",
    )
    parser.add_argument(
        "--music-volume",
        type=float,
        default=None,
        help="Background music volume in final mix (default: 0.65, or config)",
    )
    parser.add_argument(
        "--original-volume",
        type=float,
        default=None,
        help="Original clip audio volume relative to full scale (default: 0.35, or config)",
    )
    parser.add_argument(
        "--scuba-filter",
        dest="scuba_filter",
        action="store_true",
        default=None,
        help="Apply regulator-hiss filter to attenuate scuba breathing frequencies",
    )
    parser.add_argument(
        "--no-scuba-filter",
        dest="scuba_filter",
        action="store_false",
        help="Disable regulator-hiss EQ filter on original clip audio",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--trace", action="store_true", help="Enable trace logging")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional logfile path",
    )
    args = parser.parse_args()
    setup_signal_handlers()
    configure_logging(debug=args.debug, trace=args.trace, log_file=args.log_file)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    try:
        music_cfg = resolve_music_config(config, args)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    general = config.get("general", {})
    output_dir = Path(general.get("output_dir", "./outputs"))

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

    with open(detections_path, encoding="utf-8") as f:
        detections_data = json.load(f)

    detections = detections_data.get("detections", [])
    if not detections:
        logger.info("No detections found to assemble.")
        sys.exit(0)

    # Select diverse highlights with MMR and chronological ordering
    aftermovie_cfg = (
        config.get("aftermovie", {}) if isinstance(config.get("aftermovie"), dict) else {}
    )
    selection_cfg = config.get("selection", {}) if isinstance(config.get("selection"), dict) else {}
    top_n = args.top_n if args.top_n is not None else int(aftermovie_cfg.get("top_n_clips", 10))
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

    logger.info("Assembling top %d of %d detections", len(top_dets), len(detections))

    # Find rendered clip files matching each detection's clip_id
    clip_files: list[Path] = []
    for det in top_dets:
        raw_clip_id = str(det.get("clip_id", ""))
        clip_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_clip_id)[:64]
        raw_subject = str(det.get("subject", "unknown"))
        subject = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_subject)[:40]
        quality = det.get("quality_score", 0)

        start_time = float(det.get("start_time", 0))
        end_time = float(det.get("end_time", start_time + 5))

        # Look for rendered clip matching this detection (try timestamped first, then legacy)
        matching = list(
            clips_dir.glob(f"{clip_id}_{start_time:.1f}s-{end_time:.1f}s_{subject}_q{quality}.mp4")
        )
        if not matching:
            matching = list(clips_dir.glob(f"{clip_id}_{subject}_q{quality}.mp4"))
        if not matching:
            matching = list(clips_dir.glob(f"{clip_id}_{start_time:.1f}s-{end_time:.1f}s_*.mp4"))
        if matching:
            clip_files.append(matching[0])
            logger.info("  Using: %s (quality %s/10)", matching[0].name, quality)
        else:
            # Fallback: try any file starting with clip_id
            fallback = list(clips_dir.glob(f"{clip_id}_*.mp4"))
            if fallback:
                clip_files.append(fallback[0])
                logger.info(
                    "  Using: %s (quality %s/10, fallback match)", fallback[0].name, quality
                )
            else:
                logger.warning("  SKIP: No rendered clip found for %s (%s)", clip_id, subject)

    if not clip_files:
        logger.error("No clip files found to concatenate.")
        sys.exit(1)

    output_path = Path(args.output)
    try:
        if concat_clips(clip_files, output_path):
            has_cli_music = any(
                v is not None
                for v in (
                    args.atmosphere,
                    args.music,
                    args.music_volume,
                    args.original_volume,
                    args.scuba_filter,
                )
            )
            has_cfg_music = isinstance(config.get("music"), dict) or bool(
                aftermovie_cfg.get("atmosphere") or aftermovie_cfg.get("music_path")
            )
            explicit_disable = (
                args.atmosphere == "none"
                and not args.music
                and args.original_volume is None
                and args.scuba_filter is not True
            )
            should_mix = (
                not explicit_disable
                and (has_cli_music or has_cfg_music)
                and (
                    music_cfg.atmosphere != "none"
                    or music_cfg.music_path is not None
                    or args.scuba_filter is True
                    or (args.original_volume is not None and args.original_volume != 1.0)
                )
            )

            if should_mix and output_path.exists() and output_path.stat().st_size > 0:
                music_path = music_cfg.music_path
                if music_path is None and music_cfg.atmosphere != "none":
                    music_path = resolve_atmosphere_music(
                        music_cfg.atmosphere,
                        cache_dir=music_cfg.cache_dir,
                        download_enabled=music_cfg.download_enabled,
                        atmosphere_urls=music_cfg.atmosphere_urls,
                    )
                has_track = music_path is not None and music_path.exists()
                eff_orig = (
                    music_cfg.original_volume
                    if (has_track or args.original_volume is not None)
                    else None
                )
                eff_scuba = (
                    music_cfg.scuba_filter
                    if (has_track or args.scuba_filter is not None)
                    else False
                )
                eff_fade = (
                    music_cfg.audio_fade_duration
                    if (has_track or eff_orig is not None or eff_scuba)
                    else 0.0
                )
                if has_track or eff_orig is not None or eff_scuba:
                    tmp_mixed = output_path.with_name(f".tmp_mix_{output_path.name}")
                    tmp_mixed.unlink(missing_ok=True)
                    vid_dur = probe_duration(output_path)
                    if (
                        mix_video_audio(
                            output_path,
                            music_path if has_track else None,
                            tmp_mixed,
                            music_volume=music_cfg.music_volume,
                            original_volume=eff_orig,
                            scuba_filter=eff_scuba,
                            fade_duration=eff_fade,
                            video_duration=vid_dur,
                            run_fn=subprocess.run,
                        )
                        and tmp_mixed.exists()
                    ):
                        tmp_mixed.replace(output_path)
                        if has_track and music_path is not None:
                            logger.info(
                                "✓ Added background music %s (music=%s, original=%s)",
                                music_path.name,
                                music_cfg.music_volume,
                                eff_orig,
                            )
                    else:
                        tmp_mixed.unlink(missing_ok=True)
                        logger.warning("Failed to mix music; keeping unmixed assembled video.")

            logger.info("✓ Assembled %d clips into: %s", len(clip_files), output_path)
        else:
            logger.error("✗ Assembly failed.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Assembly interrupted by user (Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()
