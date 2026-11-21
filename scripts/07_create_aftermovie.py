"""Create an aftermovie with music and transitions from detection clips.

Reads detections_refined.json, sorts by quality, takes top-N clips,
renders each with fade-in/fade-out (cross-dissolve effect), concatenates
them, adds a background music track with ducking, and outputs aftermovie.mp4.
This covers Phase 2 Steps 2.2-2.4 of the simple plan.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.audio_music import (
    VALID_ATMOSPHERE_CHOICES,
    MusicConfig,
    mix_video_audio,
    resolve_atmosphere_music,
    resolve_music_config,
)
from utils.config import ConfigError, load_config
from utils.encoder import (
    VideoQuality,
    get_fallback_encoder,
    get_video_encoder,
    resolve_video_quality,
)
from utils.ffmpeg_streamer import concat_clips
from utils.logging_setup import configure_logging, get_console
from utils.models import EditPlan, StorylineProposalSet
from utils.run_archive import resolve_and_validate_assembly_artifacts, resolve_run_dir
from utils.selection import (
    select_diverse_highlights,
    select_duration_budget_highlights,
)
from utils.signals import setup_signal_handlers
from utils.storyline import load_clip_metadata, load_feedback_data

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
    bridge_gap_seconds or less (or overlap), and their combined span does not exceed
    max_clip_duration, merge them into a single continuous highlight window.
    If overlapping windows exceed max_clip_duration, clamp the start of the second
    window to avoid re-rendering duplicate frames.
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

        if prev_source == curr_source:
            gap = curr_start - prev_end
            merged_end = max(prev_end, curr_end)
            merged_duration = merged_end - prev_start

            if gap <= bridge_gap_seconds and merged_duration <= max_clip_duration:
                prev["end_time"] = merged_end
                if float(clip_copy.get("quality_score", 0)) > float(prev.get("quality_score", 0)):
                    prev["quality_score"] = clip_copy["quality_score"]
                    if "subject" in clip_copy:
                        prev["subject"] = clip_copy["subject"]
                continue

            # If not merged but still overlapping, clamp start_time to avoid duplicate rendering
            if curr_start < prev_end:
                curr_start = prev_end
                clip_copy["start_time"] = curr_start
                if curr_start >= curr_end:
                    continue

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
    original_volume: float | None = None,
    scuba_filter: bool = False,
    fade_duration: float = 0.0,
    video_duration: float | None = None,
) -> bool:
    """Mix a background music track under the video audio with optional scuba filtering.

    When ``original_volume`` and ``scuba_filter`` are left at their defaults
    (``None`` and ``False``), preserves legacy direct ``amix`` / copy behavior.
    """
    _ = aspect_ratio
    return mix_video_audio(
        video_path,
        music_path,
        output_path,
        music_volume=music_volume,
        original_volume=original_volume,
        scuba_filter=scuba_filter,
        fade_duration=fade_duration,
        video_duration=video_duration,
        run_fn=subprocess.run,
    )


def _parse_date_range(raw: str | None) -> tuple[str, str] | None:
    if not raw or not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(":", 1)]
    if len(parts) != 2:
        raise ValueError(f"Invalid --date-range {raw!r}; expected YYYY-MM-DD:YYYY-MM-DD")
    return parts[0], parts[1]


def _load_plans_from_file(
    plan_path: Path, option: str | None = None
) -> tuple[list[EditPlan], str | None]:
    """Load one or more ``EditPlan`` objects from a storyline_plans.json or EditPlan JSON."""
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("proposals"), list):
        proposal_set = StorylineProposalSet.model_validate(raw)
        proposals = proposal_set.proposals
        if not proposals:
            raise ValueError(f"No proposals found in {plan_path}")
        if option == "all":
            return proposals, proposal_set.source_detections or None
        if option:
            matched = [p for p in proposals if p.id == option]
            if not matched:
                available = ", ".join(p.id for p in proposals)
                raise ValueError(
                    f"Unknown option {option!r} in {plan_path}. Available: {available}"
                )
            return matched, proposal_set.source_detections or None
        preferred = next((p for p in proposals if p.id == "highlight"), proposals[0])
        return [preferred], proposal_set.source_detections or None

    plan = EditPlan.model_validate(raw)
    return [plan], None


def print_aftermovie_preview(
    items: list[dict[str, Any]],
    *,
    title: str = "Aftermovie Shot Preview",
    clip_meta: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print a Rich table showing exact shots, trim windows, transitions, and runtime."""
    console = get_console()
    table = Table(title=title)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Act", style="cyan")
    table.add_column("File", style="yellow")
    table.add_column("Recorded", style="dim")
    table.add_column("Window", justify="right")
    table.add_column("Dur", justify="right")
    table.add_column("Q", justify="right")
    table.add_column("Role", style="magenta")
    table.add_column("Fade", style="blue")
    table.add_column("Subject", style="green")

    total_dur = 0.0
    for idx, item in enumerate(items, start=1):
        fname = str(item.get("file_name", ""))
        ctime = str(
            item.get("creation_time") or (clip_meta or {}).get(fname, {}).get("creation_time") or ""
        )[:16].replace("T", " ")
        start_t = float(item.get("start_time", 0.0))
        end_t = float(item.get("end_time", start_t))
        dur = max(0.0, end_t - start_t)
        total_dur += dur
        f_in = "in" if item.get("fade_in") else "-"
        f_out = "out" if item.get("fade_out") else "-"
        act_str = f"Act {item['act']}" if item.get("act") else "-"
        table.add_row(
            str(idx),
            act_str,
            fname,
            ctime,
            f"{start_t:.1f}-{end_t:.1f}s",
            f"{dur:.1f}s",
            str(item.get("quality_score", "")),
            str(item.get("story_role", "")),
            f"{f_in}/{f_out}",
            str(item.get("subject", ""))[:46],
        )

    console.print(table)
    summary = (
        f"Preview: {len(items)} clips, total runtime {total_dur:.1f}s ({total_dur / 60:.2f} min)"
    )
    console.print(summary)
    logger.info("%s", summary)


def _plan_to_render_items(plan: EditPlan) -> list[dict[str, Any]]:
    return [
        {
            "clip_id": shot.clip_id,
            "file_name": shot.file_name,
            "file_path": shot.file_path,
            "creation_time": shot.creation_time,
            "act": shot.act,
            "start_time": shot.in_time,
            "end_time": shot.out_time,
            "subject": shot.subject,
            "story_role": shot.story_role,
            "quality_score": shot.quality_score,
            "fade_in": shot.transition_in == "fade",
            "fade_out": shot.transition_out == "fade",
        }
        for shot in plan.shots
    ]


def _render_shot_sequence(
    items: list[dict[str, Any]],
    *,
    clips_dir: Path,
    output_dir: Path,
    output_path: Path,
    music_path: Path | None,
    music_volume: float,
    aspect_ratio: str,
    fade_duration: float,
    video_quality: VideoQuality,
    original_volume: float | None = 0.35,
    scuba_filter: bool = True,
    audio_fade_duration: float = 1.0,
) -> bool:
    """Render a sequence of shot dicts into ``output_path``."""
    tmp_dir = output_dir / ".aftermovie_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        clip_files: list[Path] = []
        for i, det in enumerate(items):
            fade_in = bool(det.get("fade_in", True))
            fade_out = bool(det.get("fade_out", True))

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
                len(items),
                subject,
                quality_score,
            )

        if not clip_files:
            logger.error("No clip files found to assemble.")
            return False

        assembled = tmp_dir / "assembled.mp4"
        if not concat_clips(clip_files, assembled):
            logger.error("✗ Assembly failed.")
            return False

        logger.info("✓ Assembled %d clips with cross-dissolve transitions", len(clip_files))

        aspect_converted = tmp_dir / "aspect_converted.mp4"
        if aspect_ratio != "16:9":
            logger.info("Converting to aspect ratio %s...", aspect_ratio)
            if not convert_aspect_ratio(
                assembled,
                aspect_converted,
                aspect_ratio,
                quality=video_quality,
            ):
                logger.warning("Aspect ratio conversion failed, using original.")
                aspect_converted = assembled
        else:
            aspect_converted = assembled

        vid_dur = _probe_duration(aspect_converted, default=0.0)
        if add_background_music(
            aspect_converted,
            music_path,
            output_path,
            music_volume,
            aspect_ratio,
            original_volume,
            scuba_filter,
            audio_fade_duration,
            vid_dur if vid_dur > 0 else None,
        ):
            if music_path:
                logger.info(
                    "✓ Added background music %s (music=%s, original=%s, scuba_filter=%s)",
                    music_path.name,
                    music_volume,
                    original_volume,
                    scuba_filter,
                )
            logger.info("✓ Aftermovie created: %s", output_path)
        else:
            logger.warning("Failed to add music. Outputting assembled video without music.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(assembled, output_path)
            logger.info("✓ Aftermovie created (no music): %s", output_path)
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_edit_plan(
    plan: EditPlan,
    *,
    clips_dir: Path,
    output_dir: Path,
    output_path: Path,
    music_path: Path | None,
    music_volume: float,
    aspect_ratio: str,
    fade_duration: float,
    video_quality: VideoQuality,
    original_volume: float | None = 0.35,
    scuba_filter: bool = True,
    audio_fade_duration: float = 1.0,
) -> bool:
    """Render an ``EditPlan`` with explicit per-shot trim bounds and act transitions."""
    items = _plan_to_render_items(plan)
    logger.info(
        "Rendering plan %r (%d clips, ~%.1fs) -> %s",
        plan.id,
        len(items),
        plan.estimated_duration_seconds,
        output_path,
    )
    return _render_shot_sequence(
        items,
        clips_dir=clips_dir,
        output_dir=output_dir,
        output_path=output_path,
        music_path=music_path,
        music_volume=music_volume,
        aspect_ratio=aspect_ratio,
        fade_duration=fade_duration,
        video_quality=video_quality,
        original_volume=original_volume,
        scuba_filter=scuba_filter,
        audio_fade_duration=audio_fade_duration,
    )


def _resolve_aftermovie_music_mix(
    music_cfg: MusicConfig, args: argparse.Namespace
) -> tuple[Path | None, float | None, bool, float]:
    """Resolve background music path and effective scuba-audio mix settings."""
    music_path = music_cfg.music_path
    if music_path is None and music_cfg.atmosphere != "none":
        music_path = resolve_atmosphere_music(
            music_cfg.atmosphere,
            cache_dir=music_cfg.cache_dir,
            download_enabled=music_cfg.download_enabled,
            atmosphere_urls=music_cfg.atmosphere_urls,
        )
    if music_path is not None and music_path.exists():
        return (
            music_path,
            music_cfg.original_volume,
            music_cfg.scuba_filter,
            music_cfg.audio_fade_duration,
        )
    explicit_orig = getattr(args, "original_volume", None)
    explicit_scuba = getattr(args, "scuba_filter", None)
    orig_vol = float(explicit_orig) if explicit_orig is not None else None
    scuba_on = bool(explicit_scuba) if explicit_scuba is not None else False
    fade = music_cfg.audio_fade_duration if (orig_vol is not None or scuba_on) else 0.0
    return None, orig_vol, scuba_on, fade


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an aftermovie with music and transitions from detection clips"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--run",
        default=None,
        metavar="RUN_ID_OR_PATH",
        help="Run ID or path to resolve detections, highlights, and output from",
    )
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
        "--plan",
        default=None,
        help="Path to storyline_plans.json or EditPlan JSON to render",
    )
    parser.add_argument(
        "--option",
        default=None,
        help="Option ID inside storyline_plans.json to render (or 'all' to render all options)",
    )
    parser.add_argument(
        "--dry-run",
        "--preview",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Print the selected clips/timeline preview and exit without rendering",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of top-quality clips to include (default: 10, or config)",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Target total runtime in seconds (uses duration-budget selection + trimming)",
    )
    parser.add_argument(
        "--max-clip-duration",
        type=float,
        default=None,
        help="Maximum duration per trimmed clip in seconds (used with --target-duration)",
    )
    parser.add_argument(
        "--order",
        choices=["chronological", "story_arc"],
        default=None,
        help="Sequence order in duration-budget mode: chronological or story_arc",
    )
    parser.add_argument(
        "--max-per-source-file",
        type=int,
        default=None,
        help="Maximum clips allowed per source video file",
    )
    parser.add_argument(
        "--date-range",
        default=None,
        help="Inclusive recording date filter YYYY-MM-DD:YYYY-MM-DD",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=None,
        help="Source filename to exclude (can be passed multiple times)",
    )
    parser.add_argument(
        "--allow-outlier-dates",
        action="store_true",
        default=False,
        help="Disable automatic exclusion of off-trip outlier dates in duration-budget mode",
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

    general = config.get("general", {})
    output_dir = Path(general.get("output_dir", "./outputs"))
    video_quality = resolve_video_quality(config)

    aftermovie_cfg = (
        config.get("aftermovie", {}) if isinstance(config.get("aftermovie"), dict) else {}
    )
    top_n = args.top_n if args.top_n is not None else int(aftermovie_cfg.get("top_n_clips", 10))
    try:
        music_cfg = resolve_music_config(config, args)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    music_volume = music_cfg.music_volume
    fade_duration = (
        args.fade_duration
        if args.fade_duration is not None
        else float(aftermovie_cfg.get("fade_duration") or 1.0)
    )
    min_transition_gap = (
        args.min_transition_gap
        if args.min_transition_gap is not None
        else float(aftermovie_cfg.get("min_transition_gap") or 3.0)
    )
    bridge_gap_seconds = (
        args.bridge_gap_seconds
        if args.bridge_gap_seconds is not None
        else float(aftermovie_cfg.get("bridge_gap_seconds") or 1.5)
    )
    scan_params = (
        config.get("scan_parameters", {}) if isinstance(config.get("scan_parameters"), dict) else {}
    )
    max_clip_duration = float(
        aftermovie_cfg.get("max_clip_duration") or scan_params.get("max_clip_duration") or 12.0
    )

    # Plan-driven rendering path (--plan, or --option defaulting to storyline_plans.json)
    if not args.plan and args.option:
        args.plan = "storyline_plans.json"

    if args.plan:
        plan_path = Path(args.plan)
        latest_run: Path | None = None
        if args.run:
            try:
                latest_run = resolve_run_dir(output_dir, args.run)
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                sys.exit(1)
        else:
            with contextlib.suppress(FileNotFoundError):
                latest_run = resolve_run_dir(output_dir, "latest")

        if not plan_path.exists() and latest_run is not None:
            if (latest_run / plan_path).exists():
                plan_path = latest_run / plan_path
            elif plan_path.parent.name == "latest" and (latest_run / plan_path.name).exists():
                plan_path = latest_run / plan_path.name
        if not plan_path.exists() and (output_dir / plan_path).exists():
            plan_path = output_dir / plan_path

        if not plan_path.exists():
            logger.error("Plan file not found: %s", plan_path)
            sys.exit(1)
        try:
            plans, _src_dets = _load_plans_from_file(plan_path, option=args.option)
        except (OSError, ValueError, ValidationError) as exc:
            logger.error("Invalid plan file %s: %s", plan_path, exc)
            sys.exit(1)

        if latest_run is None and (plan_path.parent / "detections_refined.json").exists():
            latest_run = plan_path.parent
        clips_dir = (
            Path(args.clips_dir)
            if args.clips_dir
            else ((latest_run / "highlights") if latest_run else (output_dir / "highlights"))
        )
        clip_meta = load_clip_metadata((latest_run / "manifest.json") if latest_run else None)

        if args.dry_run:
            for plan in plans:
                print_aftermovie_preview(
                    _plan_to_render_items(plan),
                    title=f"Plan Preview: {plan.id} — {plan.title}",
                    clip_meta=clip_meta,
                )
            sys.exit(0)

        music_path, orig_vol, scuba_on, audio_fade = _resolve_aftermovie_music_mix(music_cfg, args)

        try:
            for plan in plans:
                if args.output:
                    out_p = Path(args.output)
                    if out_p.is_dir():
                        dest_path = out_p / (
                            f"aftermovie_{plan.id}.mp4" if len(plans) > 1 else "aftermovie.mp4"
                        )
                    elif len(plans) > 1 or args.option == "all":
                        dest_path = out_p.with_name(f"{out_p.stem}_{plan.id}{out_p.suffix}")
                    else:
                        dest_path = out_p
                else:
                    base_dir = latest_run if latest_run is not None else output_dir
                    dest_name = (
                        f"aftermovie_{plan.id}.mp4"
                        if (len(plans) > 1 or args.option == "all")
                        else "aftermovie.mp4"
                    )
                    dest_path = base_dir / dest_name

                if not render_edit_plan(
                    plan,
                    clips_dir=clips_dir,
                    output_dir=output_dir,
                    output_path=dest_path,
                    music_path=music_path,
                    music_volume=music_volume,
                    aspect_ratio=args.aspect_ratio,
                    fade_duration=fade_duration,
                    video_quality=video_quality,
                    original_volume=orig_vol,
                    scuba_filter=scuba_on,
                    audio_fade_duration=audio_fade,
                ):
                    sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("Aftermovie creation interrupted by user (Ctrl+C). Cleaning up...")
            sys.exit(130)
        return

    # Resolve detections input path and clips directory
    try:
        detections_path, clips_dir, latest_run = resolve_and_validate_assembly_artifacts(
            output_dir,
            input_detections=args.input,
            clips_dir=args.clips_dir,
            run=args.run,
            require_clips_dir=not args.dry_run,
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

    run_base = (
        latest_run
        if latest_run is not None and detections_path.parent.resolve() == latest_run.resolve()
        else detections_path.parent
    )
    clip_meta = load_clip_metadata(run_base / "manifest.json")
    feedback = load_feedback_data(run_base / "feedback.json")

    selection_cfg = config.get("selection", {}) if isinstance(config.get("selection"), dict) else {}
    lambda_quality = float(selection_cfg.get("lambda_quality", 0.65))
    max_per_clip = (
        int(selection_cfg["per_clip_soft_limit"])
        if selection_cfg.get("per_clip_soft_limit") is not None
        else None
    )

    try:
        date_range = _parse_date_range(args.date_range)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Duration-budget / story-arc mode when --target-duration or --order is requested
    if args.target_duration is not None or args.order is not None:
        target_dur = float(args.target_duration if args.target_duration is not None else 150.0)
        trim_cap = float(args.max_clip_duration if args.max_clip_duration is not None else 6.5)
        max_per_source = (
            args.max_per_source_file
            if args.max_per_source_file is not None
            else int(selection_cfg.get("max_per_source_file", 1))
        )
        outlier_days = int(selection_cfg.get("outlier_days", 7))
        order_mode = args.order or "story_arc"
        plan, _filter_report = select_duration_budget_highlights(
            detections,
            clip_meta=clip_meta,
            feedback=feedback,
            target_duration=target_dur,
            clip_duration_cap=trim_cap,
            lambda_quality=lambda_quality,
            max_per_clip=max_per_clip if max_per_clip is not None else 1,
            max_per_source_file=max_per_source,
            order=order_mode,
            outlier_days=outlier_days,
            allow_outlier_dates=args.allow_outlier_dates,
            date_range=date_range,
            exclude_files=args.exclude_file,
        )
        items = _plan_to_render_items(plan)
        if args.dry_run:
            print_aftermovie_preview(
                items,
                title=f"Deterministic Preview ({order_mode}, target={target_dur:.0f}s)",
                clip_meta=clip_meta,
            )
            sys.exit(0)

        if args.output:
            output_path = Path(args.output)
        elif latest_run is not None:
            output_path = latest_run / "aftermovie.mp4"
        else:
            output_path = output_dir / "aftermovie.mp4"

        music_path, orig_vol, scuba_on, audio_fade = _resolve_aftermovie_music_mix(music_cfg, args)

        try:
            if not render_edit_plan(
                plan,
                clips_dir=clips_dir,
                output_dir=output_dir,
                output_path=output_path,
                music_path=music_path,
                music_volume=music_volume,
                aspect_ratio=args.aspect_ratio,
                fade_duration=fade_duration,
                video_quality=video_quality,
                original_volume=orig_vol,
                scuba_filter=scuba_on,
                audio_fade_duration=audio_fade,
            ):
                sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("Aftermovie creation interrupted by user (Ctrl+C). Cleaning up...")
            sys.exit(130)
        return

    # Classic --top-n MMR selection + chronological ordering
    trace_path = (
        (latest_run / "selection_trace.json")
        if latest_run is not None
        else (output_dir / "selection_trace.json")
    )
    select_kwargs: dict[str, Any] = {
        "top_n": top_n,
        "lambda_quality": lambda_quality,
        "max_per_clip": max_per_clip,
        "trace_path": trace_path,
    }
    if clip_meta:
        select_kwargs["clip_meta"] = clip_meta
    if args.max_per_source_file is not None:
        select_kwargs["max_per_source_file"] = args.max_per_source_file

    top_dets = select_diverse_highlights(detections, **select_kwargs)

    top_dets = bridge_clip_gaps(
        top_dets,
        bridge_gap_seconds=bridge_gap_seconds,
        max_clip_duration=max_clip_duration,
    )
    transitions = compute_clip_transitions(
        top_dets,
        min_transition_gap=min_transition_gap,
    )
    render_items = [
        {**det, "fade_in": transitions[i]["fade_in"], "fade_out": transitions[i]["fade_out"]}
        for i, det in enumerate(top_dets)
    ]

    if args.dry_run:
        print_aftermovie_preview(
            render_items,
            title=f"Top-{top_n} Preview",
            clip_meta=clip_meta,
        )
        sys.exit(0)

    logger.info(
        "Creating aftermovie from %d highlight clips (from top %d detections)",
        len(top_dets),
        min(top_n, len(detections)),
    )

    if args.output:
        output_path = Path(args.output)
    elif latest_run is not None:
        output_path = latest_run / "aftermovie.mp4"
    else:
        output_path = output_dir / "aftermovie.mp4"

    music_path, orig_vol, scuba_on, audio_fade = _resolve_aftermovie_music_mix(music_cfg, args)

    try:
        if not _render_shot_sequence(
            render_items,
            clips_dir=clips_dir,
            output_dir=output_dir,
            output_path=output_path,
            music_path=music_path,
            music_volume=music_volume,
            aspect_ratio=args.aspect_ratio,
            fade_duration=fade_duration,
            video_quality=video_quality,
            original_volume=orig_vol,
            scuba_filter=scuba_on,
            audio_fade_duration=audio_fade,
        ):
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Aftermovie creation interrupted by user (Ctrl+C). Cleaning up...")
        sys.exit(130)


if __name__ == "__main__":
    main()
