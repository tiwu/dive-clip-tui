"""Coarse-to-fine VLM scanner for underwater footage.

Pass 1: Coarse scan at 0.5 FPS - identify candidate windows.
Pass 2: Fine scan at 2.0 FPS around candidates with confidence >= 0.65.
Uses Qwen2.5-VL via Ollama with JSON mode.
Includes error recovery for corrupt frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.checkpoint import PipelineState
from utils.config import ConfigError, load_config
from utils.ffmpeg_streamer import cleanup_work_dir, extract_frames, get_work_dir
from utils.logging_setup import configure_logging, get_console
from utils.models import DetectionsRefined, RefinedDetection
from utils.ollama_client import OllamaClient, resolve_host
from utils.prompt_templates import (
    BROAD_MARINE_TEMPLATE,
    get_template_for_keywords,
)
from utils.run_archive import archive_run_artifacts, get_run_dir
from utils.stats import (
    Phase1Stats,
    Phase2Stats,
    PipelineStats,
    format_stats_log,
    format_stats_table,
    parse_duration,
    save_stats_json,
)

logger = logging.getLogger(__name__)


def setup_signal_handlers() -> None:
    """Ensure SIGINT and SIGTERM raise KeyboardInterrupt for clean shutdown."""

    def _handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except ValueError, AttributeError:
        pass


def call_ollama(
    host: str,
    model: str,
    image_path: str,
    prompt: str,
    timeout: int = 90,
    *,
    client: OllamaClient | None = None,
    phase: str = "unspecified",
) -> dict[str, Any] | None:
    """Compatibility wrapper around the shared Ollama client."""
    request_client = client or OllamaClient(host, model, timeout)
    return request_client.generate(image_path, prompt, phase)


def _summarize_scores(scores: list[float]) -> str:
    """Format min/avg/max of confidence scores for log output."""
    if not scores:
        return "n/a"
    return f"{min(scores):.2f}/{sum(scores) / len(scores):.2f}/{max(scores):.2f}"


class StatusColumn(ProgressColumn):
    """Renders optional status field safely."""

    def render(self, task: Any) -> Text:
        status = task.fields.get("status", "")
        return Text(f" {status}", style="dim") if status else Text("")


def create_scan_progress(console: Console | None = None) -> Progress:
    """Create a configured Rich Progress instance for video scanning."""
    active_console = console or get_console()
    return Progress(
        SpinnerColumn(),
        TextColumn(
            "{task.description}",
            table_column=Column(no_wrap=True, overflow="ellipsis", max_width=32),
        ),
        BarColumn(bar_width=16),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        StatusColumn(table_column=Column(no_wrap=True, max_width=16)),
        console=active_console,
        disable=not active_console.is_terminal,
    )


def process_clip(
    clip: dict,
    config: dict,
    work_dir: Path,
    ollama_client: OllamaClient | None = None,
    state: PipelineState | None = None,
    progress: Progress | None = None,
    phase1_stats: Phase1Stats | None = None,
    phase2_stats: Phase2Stats | None = None,
) -> list[RefinedDetection]:
    """Process a single clip through coarse-to-fine scanning.

    Returns a list of refined detections (may be empty).
    """
    file_path = clip["file_path"]
    if not Path(file_path).exists():
        logger.warning("File not found: %s", file_path)
        return []

    raw_clip_id = clip.get("clip_id") or (
        f"clip_{hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:16]}"
    )
    clip_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_clip_id)[:64]
    clip_work_dir = work_dir / clip_id
    clip_work_dir.mkdir(parents=True, exist_ok=True)

    scan_params = config.get("scan_parameters", {})
    coarse_fps = scan_params.get("coarse_fps", 0.5)
    coarse_scale = scan_params.get("coarse_scale", "640:-2")
    fine_fps = scan_params.get("fine_fps", 2.0)
    fine_scale = scan_params.get("fine_scale")
    coarse_threshold = scan_params.get("coarse_confidence_threshold", 0.65)
    coarse_quality_threshold = scan_params.get("coarse_quality_threshold", 4)
    fine_padding = scan_params.get("fine_padding_seconds", 4)
    quality_threshold = scan_params.get("quality_threshold", 7)

    ollama_cfg = config.get("ollama", {})
    ollama_host = resolve_host(ollama_cfg.get("host"))
    ollama_model = ollama_cfg.get("model", "qwen2.5-vl:7b")
    ollama_timeout = ollama_cfg.get("timeout_seconds", 90)
    client = ollama_client or OllamaClient(ollama_host, ollama_model, ollama_timeout)

    tags = clip.get("tags", [])

    # ---------- Pass 1: Coarse scan ----------
    logger.info("[Pass 1] Coarse scan @ %.1f FPS: %s", coarse_fps, clip["file_name"])
    if state is not None:
        state.set_activity(f"scan: {clip['file_name']} (pass 1 coarse, {coarse_fps} FPS)")

    coarse_frames = extract_frames(
        video_path=file_path,
        output_dir=clip_work_dir / "coarse",
        fps=coarse_fps,
        scale=coarse_scale,
    )
    if phase1_stats is not None:
        phase1_stats.frames_extracted += len(coarse_frames)
    logger.info("[Pass 1] Extracted %d frames from %s", len(coarse_frames), clip["file_name"])

    template = get_template_for_keywords(tags)
    coarse_prompt = template.coarse_prompt if template else BROAD_MARINE_TEMPLATE.coarse_prompt

    candidate_windows: list[tuple[float, float]] = []
    coarse_confidences: list[float] = []
    coarse_detected = 0
    coarse_failures = 0
    coarse_elapsed_total = 0.0

    task_id = None
    if progress is not None and coarse_frames:
        task_id = progress.add_task(
            f"[cyan]Pass 1:[/] {clip['file_name']}",
            total=len(coarse_frames),
            status="(0 hits)",
        )

    try:
        for frame_index, frame in enumerate(coarse_frames, start=1):
            if phase1_stats is not None:
                phase1_stats.frames_analyzed += 1
            logger.debug("[Pass 1] Analyzing frame %d of %d", frame_index, len(coarse_frames))

            start_time = time.perf_counter()
            result = call_ollama(
                ollama_host,
                ollama_model,
                frame["frame_path"],
                coarse_prompt,
                ollama_timeout,
                client=client,
                phase="coarse",
            )
            elapsed = time.perf_counter() - start_time
            coarse_elapsed_total += elapsed

            if result is None:
                coarse_failures += 1
                logger.debug(
                    "[Pass 1] Frame %d @ %.1fs: no result (%.1fs)",
                    frame_index,
                    frame["timestamp"],
                    elapsed,
                )
                if progress is not None and task_id is not None:
                    progress.advance(task_id)
                continue

            detected = bool(result.get("detected"))
            confidence = float(result.get("confidence", 0.0) or 0.0)
            quality_score = int(result.get("quality_score", 0) or 0)
            coarse_confidences.append(confidence)
            if detected:
                coarse_detected += 1
                if phase1_stats is not None:
                    phase1_stats.detections_detected += 1

            frame_passes = (
                detected
                and confidence >= coarse_threshold
                and quality_score >= coarse_quality_threshold
            )
            if frame_passes:
                ts = frame["timestamp"]
                candidate_windows.append((max(0, ts - fine_padding), ts + fine_padding))

            if progress is not None and task_id is not None:
                progress.advance(task_id)
                progress.update(task_id, status=f"({len(candidate_windows)} hits)")

            logger.debug(
                "[Pass 1] Frame %d @ %.1fs: detected=%s confidence=%.2f quality=%d passes_threshold=%s "
                "subject=%s (%.1fs)",
                frame_index,
                frame["timestamp"],
                detected,
                confidence,
                quality_score,
                frame_passes,
                result.get("subject", "-"),
                elapsed,
            )
    finally:
        if progress is not None and task_id is not None:
            progress.remove_task(task_id)

    logger.info(
        "[Pass 1] Summary %s: %d frames, %d detected, %d over threshold %.2f, "
        "confidence min/avg/max = %s, %d failed calls, %.1fs total",
        clip["file_name"],
        len(coarse_frames),
        coarse_detected,
        len(candidate_windows),
        coarse_threshold,
        _summarize_scores(coarse_confidences),
        coarse_failures,
        coarse_elapsed_total,
    )

    if not candidate_windows:
        logger.info("[Pass 1] No candidates found for %s", clip["file_name"])
        if phase1_stats is not None:
            phase1_stats.elapsed_seconds += coarse_elapsed_total
        cleanup_work_dir(clip_work_dir)
        return []

    # Merge overlapping windows
    candidate_windows.sort()
    merged: list[tuple[float, float]] = []
    for start, end in candidate_windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    if phase1_stats is not None:
        phase1_stats.candidate_windows += len(merged)
        phase1_stats.candidate_duration += sum(end - start for start, end in merged)
        phase1_stats.elapsed_seconds += coarse_elapsed_total

    # ---------- Pass 2: Fine scan ----------
    logger.info(
        "[Pass 2] Fine scan @ %.1f FPS: %s (%d windows)", fine_fps, clip["file_name"], len(merged)
    )
    if phase2_stats is not None:
        phase2_stats.windows_scanned += len(merged)
    all_detections: list[RefinedDetection] = []
    fine_frames_analyzed = 0
    fine_passed = 0
    fine_failures = 0
    fine_elapsed_total = 0.0
    fine_confidences: list[float] = []

    template = get_template_for_keywords(tags)
    fine_prompt = template.fine_prompt if template else BROAD_MARINE_TEMPLATE.fine_prompt
    color = template.color if template else BROAD_MARINE_TEMPLATE.color

    for window_index, (win_start, win_end) in enumerate(merged, start=1):
        duration = win_end - win_start
        win_dir = clip_work_dir / "fine" / f"win_{int(win_start)}_{int(win_end)}"
        logger.info(
            "[Pass 2] Window %d of %d: %.1fs-%.1fs (%.1fs) @ %.1f FPS",
            window_index,
            len(merged),
            win_start,
            win_end,
            duration,
            fine_fps,
        )
        if state is not None:
            state.set_activity(
                f"scan: {clip['file_name']} window {window_index}/{len(merged)} @{win_start:.0f}s"
            )

        fine_frames = extract_frames(
            video_path=file_path,
            output_dir=win_dir,
            fps=fine_fps,
            scale=fine_scale,
            start_seconds=win_start,
            duration_seconds=duration,
        )
        if phase2_stats is not None:
            phase2_stats.frames_extracted += len(fine_frames)
        logger.info(
            "[Pass 2] Window %d of %d: extracted %d frames",
            window_index,
            len(merged),
            len(fine_frames),
        )
        fine_frames_analyzed += len(fine_frames)

        fine_task_id = None
        if progress is not None and fine_frames:
            fine_task_id = progress.add_task(
                f"[green]Pass 2 (win {window_index}/{len(merged)}):[/] {clip['file_name']}",
                total=len(fine_frames),
                status=f"({fine_passed} hits)",
            )

        try:
            for frame_index, frame in enumerate(fine_frames, start=1):
                if phase2_stats is not None:
                    phase2_stats.frames_analyzed += 1
                logger.debug(
                    "[Pass 2] Window %d of %d: analyzing frame %d of %d",
                    window_index,
                    len(merged),
                    frame_index,
                    len(fine_frames),
                )
                start_time = time.perf_counter()
                result = call_ollama(
                    ollama_host,
                    ollama_model,
                    frame["frame_path"],
                    fine_prompt,
                    ollama_timeout,
                    client=client,
                    phase="fine",
                )
                elapsed = time.perf_counter() - start_time
                fine_elapsed_total += elapsed

                if result is None:
                    fine_failures += 1
                    logger.debug(
                        "[Pass 2] Frame @ %.2fs: no result (%.1fs)", frame["timestamp"], elapsed
                    )
                    if progress is not None and fine_task_id is not None:
                        progress.advance(fine_task_id)
                    continue

                detected = bool(result.get("detected"))
                confidence = float(result.get("confidence", 0.0) or 0.0)
                quality_score = int(result.get("quality_score", 0) or 0)
                frame_passes = detected and quality_score >= quality_threshold
                fine_confidences.append(confidence)
                if frame_passes:
                    fine_passed += 1
                    if phase2_stats is not None:
                        phase2_stats.detections_found += 1
                        ts = frame["timestamp"]
                        phase2_stats.detections_duration += (ts + 1.0) - max(0.0, ts - 1.0)

                if progress is not None and fine_task_id is not None:
                    progress.advance(fine_task_id)
                    progress.update(fine_task_id, status=f"({fine_passed} hits)")

                logger.debug(
                    "[Pass 2] Frame @ %.2fs: detected=%s confidence=%.2f quality=%d "
                    "passes_threshold=%s subject=%s (%.1fs)",
                    frame["timestamp"],
                    detected,
                    confidence,
                    quality_score,
                    frame_passes,
                    result.get("subject", "-"),
                    elapsed,
                )

                if not frame_passes:
                    continue

                try:
                    all_detections.append(
                        RefinedDetection(
                            clip_id=clip_id,
                            file_name=clip["file_name"],
                            file_path=file_path,
                            fps=str(clip.get("fps") or "25.0"),
                            start_time=max(0, frame["timestamp"] - 1),
                            end_time=frame["timestamp"] + 1,
                            subject=result.get("subject", "unknown"),
                            confidence=confidence,
                            quality_score=quality_score,
                            description=result.get("description", ""),
                            color=color,
                            template=template.name if template else "generic",
                        )
                    )
                except (ValueError, TypeError) as exc:
                    logger.warning("Skipping invalid detection: %s", exc)
                    continue
        finally:
            if progress is not None and fine_task_id is not None:
                progress.remove_task(fine_task_id)

    # Deduplicate overlapping detections (keep highest quality_score)
    before_dedup = len(all_detections)
    all_detections = _deduplicate_detections(all_detections)
    if phase2_stats is not None:
        phase2_stats.detections_dedup += len(all_detections)
        phase2_stats.detections_dedup_duration += sum(
            d.end_time - d.start_time for d in all_detections
        )
        phase2_stats.elapsed_seconds += fine_elapsed_total
    logger.info(
        "[Pass 2] Summary %s: %d windows, %d frames analyzed, %d passed quality>=%d "
        "(conf %s), %d failed calls, %d -> %d detections after dedup, %.1fs total",
        clip["file_name"],
        len(merged),
        fine_frames_analyzed,
        fine_passed,
        quality_threshold,
        _summarize_scores(fine_confidences),
        fine_failures,
        before_dedup,
        len(all_detections),
        fine_elapsed_total,
    )

    cleanup_work_dir(clip_work_dir)
    return all_detections


def _deduplicate_detections(
    detections: list[RefinedDetection],
) -> list[RefinedDetection]:
    """Merge contiguous detections of the same subject and deduplicate overlapping windows.

    1. Contiguous or adjacent windows of the same subject in the same clip are merged
       into a continuous highlight segment, preserving the highest quality score,
       highest confidence, and most descriptive detail.
    2. Any remaining overlapping windows of different subjects are deduplicated by
       keeping the detection with the higher quality score.
    """
    if not detections:
        return []

    # Step 1: Group by clip and subject (case-insensitive) to merge contiguous segments
    from collections import defaultdict

    groups: dict[tuple[str, str], list[RefinedDetection]] = defaultdict(list)
    for det in detections:
        key = (det.clip_id, det.subject.strip().lower())
        groups[key].append(det)

    all_merged: list[RefinedDetection] = []
    for group in groups.values():
        sorted_group = sorted(group, key=lambda d: d.start_time)
        merged_group: list[RefinedDetection] = []
        for det in sorted_group:
            if not merged_group:
                merged_group.append(det)
                continue
            prev = merged_group[-1]
            # Merge if overlapping or adjacent (within 1.0 second tolerance)
            if det.start_time <= prev.end_time + 1.0:
                new_end = max(prev.end_time, det.end_time)
                new_quality = max(prev.quality_score, det.quality_score)
                new_confidence = max(prev.confidence, det.confidence)
                new_desc = (
                    det.description if det.quality_score > prev.quality_score else prev.description
                )
                new_subject = (
                    det.subject if det.quality_score > prev.quality_score else prev.subject
                )
                merged_group[-1] = prev.model_copy(
                    update={
                        "end_time": new_end,
                        "quality_score": new_quality,
                        "confidence": new_confidence,
                        "description": new_desc,
                        "subject": new_subject,
                    }
                )
            else:
                merged_group.append(det)
        all_merged.extend(merged_group)

    # Step 2: For any remaining overlapping detections across different subjects, keep higher quality
    sorted_dets = sorted(all_merged, key=lambda d: (-d.quality_score, -d.confidence, d.start_time))
    kept: list[RefinedDetection] = []

    for det in sorted_dets:
        overlaps = any(
            det.clip_id == kept_det.clip_id
            and det.start_time < kept_det.end_time
            and det.end_time > kept_det.start_time
            for kept_det in kept
        )
        if not overlaps:
            kept.append(det)

    kept.sort(key=lambda d: (d.file_name, d.start_time))
    return kept


def save_detections_atomic(output_path: Path, refined: DetectionsRefined) -> None:
    """Atomically write detections to disk using a temporary file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(refined.model_dump(), f, indent=2, ensure_ascii=False)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coarse-to-fine VLM scanner for underwater footage"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override general.output_dir for manifest input and detections output",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Pipeline state file to heartbeat (default: ./outputs/.pipeline_state.json)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append this run's log lines to the given file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume scanning from last saved state, skipping completed clips",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest JSON file (default: <output-dir>/manifest.json)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory to archive detections and logs into",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive detections into a new timestamped directory under <output>/runs/run-<timestamp>/",
    )
    args = parser.parse_args()

    configure_logging(debug=args.debug, log_file=args.log_file)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.get("general", {}).get("output_dir", "./outputs"))

    if args.manifest is not None:
        manifest_path = args.manifest
    else:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = output_dir / "manifest_raw.json"

        if not manifest_path.exists() and args.output_dir is None:
            # Legacy fallback for manifests written before output_dir was configured.
            alt = Path(__file__).resolve().parent.parent / "outputs" / "manifest.json"
            if alt.exists():
                manifest_path = alt
            else:
                manifest_path = (
                    Path(__file__).resolve().parent.parent / "outputs" / "manifest_raw.json"
                )

    if not manifest_path.exists():
        logger.error("manifest.json not found at %s", manifest_path)
        logger.error("Run 00_ingest_media.py --input <folder> to generate it,")
        logger.error("or 01_extract_manifest.py from inside DaVinci Resolve.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    scan_params = config.get("scan_parameters", {})
    processing_cfg = config.get("processing", {})
    use_ramdisk = bool(
        scan_params.get(
            "use_ramdisk",
            processing_cfg.get("use_ramdisk", True),
        )
    )
    work_dir = get_work_dir(use_ramdisk=use_ramdisk)
    logger.info("Using work directory: %s (ramdisk=%s)", work_dir, use_ramdisk)
    setup_signal_handlers()

    ollama_cfg = config.get("ollama", {})
    ollama_client = OllamaClient(
        resolve_host(ollama_cfg.get("host")),
        ollama_cfg.get("model", "qwen2.5-vl:7b"),
        ollama_cfg.get("timeout_seconds", 90),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detections_refined.json"

    target_run_dir = args.run_dir
    if target_run_dir is None and args.archive:
        target_run_dir = get_run_dir(output_dir)

    refined = DetectionsRefined(total_clips=0, total_detections=0, detections=[])
    completed_clips: set[str] = set()

    state: PipelineState | None = None
    if args.state_path is not None:
        state = PipelineState(args.state_path)
        state.mark_step_started("scan")

    if args.resume and output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                loaded_data = json.load(f)
            loaded_refined = DetectionsRefined.model_validate(loaded_data)
            refined = loaded_refined
            for det in refined.detections:
                completed_clips.add(det.file_name)
            logger.info(
                "Resumed from %s with %d existing detections across %d clips",
                output_path,
                refined.total_detections,
                refined.total_clips,
            )
        except Exception as exc:
            logger.warning("Could not load existing detections from %s: %s", output_path, exc)

    clips = manifest.get("clips", [])
    if args.resume and state is not None:
        for clip in clips:
            cname = clip.get("file_name", "unknown")
            if state.is_item_done("scan", cname):
                completed_clips.add(cname)

    scan_start_time = time.perf_counter()
    p1_stats = Phase1Stats()
    p2_stats = Phase2Stats()
    if args.resume and state is not None:
        prev_scan_stats = state.get("scan_stats")
        if isinstance(prev_scan_stats, dict):
            try:
                prev_p1 = prev_scan_stats.get("phase1")
                prev_p2 = prev_scan_stats.get("phase2")
                if isinstance(prev_p1, dict):
                    p1_stats = Phase1Stats.model_validate(prev_p1)
                if isinstance(prev_p2, dict):
                    p2_stats = Phase2Stats.model_validate(prev_p2)
            except Exception as exc:
                logger.debug("Could not restore previous scan_stats: %s", exc)

    try:
        with create_scan_progress() as progress:
            overall_task = (
                progress.add_task("[bold white]Overall clips[/]", total=len(clips), status="")
                if len(clips) > 1
                else None
            )
            for index, clip in enumerate(clips, start=1):
                clip_name = clip.get("file_name", "unknown")
                if args.resume and clip_name in completed_clips:
                    logger.info(
                        "Skipping already processed clip: %s (%d/%d)", clip_name, index, len(clips)
                    )
                    if overall_task is not None:
                        progress.advance(overall_task)
                    continue

                logger.info("Processing %d/%d: %s", index, len(clips), clip_name)
                if state is not None:
                    state.set_activity(f"scan: starting {clip_name} ({index}/{len(clips)})")
                detections = process_clip(
                    clip,
                    config,
                    work_dir,
                    ollama_client,
                    state=state,
                    progress=progress,
                    phase1_stats=p1_stats,
                    phase2_stats=p2_stats,
                )
                refined.detections.extend(detections)
                refined.total_detections += len(detections)
                refined.total_clips += 1
                save_detections_atomic(output_path, refined)
                if overall_task is not None:
                    progress.advance(overall_task)
                if state is not None:
                    state.add_progress("scan", clip_name)

        # Final atomic write to ensure complete sync
        save_detections_atomic(output_path, refined)

        # Record Ollama calls for each phase
        coarse_stats = ollama_client.metrics.by_phase.get("coarse")
        if coarse_stats:
            p1_stats.ollama_calls += coarse_stats.calls
            p1_stats.ollama_failures += coarse_stats.failures
        fine_stats = ollama_client.metrics.by_phase.get("fine")
        if fine_stats:
            p2_stats.ollama_calls += fine_stats.calls
            p2_stats.ollama_failures += fine_stats.failures

        total_video_duration = sum(parse_duration(clip.get("duration")) for clip in clips)
        scan_elapsed = time.perf_counter() - scan_start_time

        scan_pipeline_stats = PipelineStats(
            run_id=target_run_dir.name if target_run_dir else "",
            video_count=len(clips),
            total_video_duration=round(total_video_duration, 2),
            total_elapsed_seconds=round(scan_elapsed, 2),
            phase1=p1_stats,
            phase2=p2_stats,
            step_durations={"scan": round(scan_elapsed, 2)},
        )

        if state is not None:
            state.set("scan_stats", scan_pipeline_stats.to_dict())
            state.save()

        stats_save_path = (
            (target_run_dir / "run_stats.json")
            if target_run_dir
            else (output_path.parent / "run_stats.json")
        )
        save_stats_json(scan_pipeline_stats, stats_save_path)

        if target_run_dir is not None:
            archive_run_artifacts(
                target_run_dir,
                manifest_path=manifest_path,
                detections_path=output_path,
                log_file=args.log_file,
                stats_path=stats_save_path,
            )
            logger.info("Archived run artifacts to %s", target_run_dir)

        console = get_console()
        if console.is_terminal:
            console.print(format_stats_table(scan_pipeline_stats, title="Scan Summary"))

        logger.info(
            "Done. %d detections across %d clips.", refined.total_detections, refined.total_clips
        )
        logger.info(ollama_client.metrics.format_summary())
        logger.info(format_stats_log(scan_pipeline_stats))
        logger.info("Output: %s", output_path)

    except KeyboardInterrupt:
        logger.warning("\nScan interrupted by user (Ctrl+C). Cleaning up...")
        if refined.detections:
            save_detections_atomic(output_path, refined)
            logger.info("Saved %d detections before exit.", len(refined.detections))
            if target_run_dir is not None:
                try:
                    archive_run_artifacts(
                        target_run_dir,
                        manifest_path=manifest_path,
                        detections_path=output_path,
                        log_file=args.log_file,
                    )
                except Exception as exc:
                    logger.warning("Could not archive partial run artifacts on interrupt: %s", exc)
        if state is not None:
            state.set_activity("scan: interrupted by user")
        sys.exit(130)
    finally:
        cleanup_work_dir(work_dir)


if __name__ == "__main__":
    main()
