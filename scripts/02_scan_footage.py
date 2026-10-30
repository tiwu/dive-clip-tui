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
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.checkpoint import PipelineState
from utils.ffmpeg_streamer import cleanup_work_dir, extract_frames, get_work_dir
from utils.logging_setup import configure_logging
from utils.models import DetectionsRefined, RefinedDetection
from utils.ollama_client import OllamaClient, resolve_host
from utils.prompt_templates import (
    BROAD_MARINE_TEMPLATE,
    get_template_for_keywords,
)

logger = logging.getLogger(__name__)


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


def process_clip(
    clip: dict,
    config: dict,
    work_dir: Path,
    ollama_client: OllamaClient | None = None,
    state: PipelineState | None = None,
) -> list[RefinedDetection]:
    """Process a single clip through coarse-to-fine scanning.

    Returns a list of refined detections (may be empty).
    """
    file_path = clip["file_path"]
    if not Path(file_path).exists():
        logger.warning("File not found: %s", file_path)
        return []

    clip_id = clip.get("clip_id") or (
        f"clip_{hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:16]}"
    )
    clip_work_dir = work_dir / clip_id
    clip_work_dir.mkdir(parents=True, exist_ok=True)

    scan_params = config.get("scan_parameters", {})
    coarse_fps = scan_params.get("coarse_fps", 0.5)
    fine_fps = scan_params.get("fine_fps", 2.0)
    coarse_threshold = scan_params.get("coarse_confidence_threshold", 0.65)
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
    )
    logger.info("[Pass 1] Extracted %d frames from %s", len(coarse_frames), clip["file_name"])

    template = get_template_for_keywords(tags)
    prompt_template = template.user_prompt if template else BROAD_MARINE_TEMPLATE.user_prompt

    candidate_windows: list[tuple[float, float]] = []
    coarse_confidences: list[float] = []
    coarse_detected = 0
    coarse_failures = 0
    coarse_elapsed_total = 0.0

    for frame_index, frame in enumerate(coarse_frames, start=1):
        logger.debug("[Pass 1] Analyzing frame %d of %d", frame_index, len(coarse_frames))

        start_time = time.perf_counter()
        result = call_ollama(
            ollama_host,
            ollama_model,
            frame["frame_path"],
            prompt_template,
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
            continue

        detected = bool(result.get("detected"))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        coarse_confidences.append(confidence)
        if detected:
            coarse_detected += 1

        frame_passes = detected and confidence >= coarse_threshold
        if frame_passes:
            ts = frame["timestamp"]
            candidate_windows.append((max(0, ts - fine_padding), ts + fine_padding))

        logger.debug(
            "[Pass 1] Frame %d @ %.1fs: detected=%s confidence=%.2f passes_threshold=%s "
            "subject=%s (%.1fs)",
            frame_index,
            frame["timestamp"],
            detected,
            confidence,
            frame_passes,
            result.get("subject", "-"),
            elapsed,
        )

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

    # ---------- Pass 2: Fine scan ----------
    logger.info(
        "[Pass 2] Fine scan @ %.1f FPS: %s (%d windows)", fine_fps, clip["file_name"], len(merged)
    )
    all_detections: list[RefinedDetection] = []
    fine_frames_analyzed = 0
    fine_passed = 0
    fine_failures = 0
    fine_elapsed_total = 0.0
    fine_confidences: list[float] = []

    template = get_template_for_keywords(tags)
    prompt_template = template.user_prompt if template else BROAD_MARINE_TEMPLATE.user_prompt
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
            start_seconds=win_start,
            duration_seconds=duration,
        )
        logger.info(
            "[Pass 2] Window %d of %d: extracted %d frames",
            window_index,
            len(merged),
            len(fine_frames),
        )
        fine_frames_analyzed += len(fine_frames)

        for frame_index, frame in enumerate(fine_frames, start=1):
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
                prompt_template,
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
                continue

            detected = bool(result.get("detected"))
            confidence = float(result.get("confidence", 0.0) or 0.0)
            quality_score = int(result.get("quality_score", 0) or 0)
            frame_passes = detected and quality_score >= quality_threshold
            fine_confidences.append(confidence)
            if frame_passes:
                fine_passed += 1

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

    # Deduplicate overlapping detections (keep highest quality_score)
    before_dedup = len(all_detections)
    all_detections = _deduplicate_detections(all_detections)
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
    """Merge temporally overlapping detections, keeping the highest quality score.

    Two detections overlap if their time windows intersect. When they overlap,
    the one with the higher quality_score is kept and the other is dropped.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: (-d.quality_score, d.start_time))
    kept: list[RefinedDetection] = []

    for det in sorted_dets:
        overlaps = any(
            det.start_time < kept_det.end_time and det.end_time > kept_det.start_time
            for kept_det in kept
        )
        if not overlaps:
            kept.append(det)

    return kept


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
    args = parser.parse_args()

    configure_logging(debug=args.debug, log_file=args.log_file)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.get("general", {}).get("output_dir", "./outputs"))
    manifest_path = output_dir / "manifest_raw.json"

    if not manifest_path.exists() and args.output_dir is None:
        # Legacy fallback for manifests written before output_dir was configured.
        manifest_path = Path(__file__).resolve().parent.parent / "outputs" / "manifest_raw.json"

    if not manifest_path.exists():
        logger.error("manifest_raw.json not found at %s", manifest_path)
        logger.error("Run 00_ingest_media.py --input <folder> to generate it,")
        logger.error("or 01_extract_manifest.py from inside DaVinci Resolve.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    work_dir = get_work_dir()
    logger.info("Using work directory: %s", work_dir)

    ollama_cfg = config.get("ollama", {})
    ollama_client = OllamaClient(
        resolve_host(ollama_cfg.get("host")),
        ollama_cfg.get("model", "qwen2.5-vl:7b"),
        ollama_cfg.get("timeout_seconds", 90),
    )
    refined = DetectionsRefined(total_clips=0, total_detections=0, detections=[])

    state: PipelineState | None = None
    if args.state_path is not None:
        state = PipelineState(args.state_path)
        state.mark_step_started("scan")

    clips = manifest.get("clips", [])
    for index, clip in enumerate(clips, start=1):
        clip_name = clip.get("file_name", "unknown")
        logger.info("Processing %d/%d: %s", index, len(clips), clip_name)
        if state is not None:
            state.set_activity(f"scan: starting {clip_name} ({index}/{len(clips)})")
        detections = process_clip(clip, config, work_dir, ollama_client, state=state)
        refined.detections.extend(detections)
        refined.total_detections += len(detections)
        refined.total_clips += 1
        if state is not None:
            state.add_progress("scan", clip_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detections_refined.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined.model_dump(), f, indent=2, ensure_ascii=False)

    # Clean up the top-level work directory (only removes mkdtemp dirs on macOS;
    # /dev/shm/dive_worker_tmp on Linux is intentionally skipped)
    cleanup_work_dir(work_dir)

    logger.info(
        "Done. %d detections across %d clips.", refined.total_detections, refined.total_clips
    )
    logger.info(ollama_client.metrics.format_summary())
    logger.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
