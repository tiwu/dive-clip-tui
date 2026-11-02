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
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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

from utils.checkpoint import (
    ClipScanCheckpoint,
    CoarseFrameResult,
    PipelineState,
    remove_clip_checkpoint,
)
from utils.config import ConfigError, load_config
from utils.ffmpeg_streamer import (
    calculate_sample_timestamps,
    cleanup_work_dir,
    extract_frames,
    extract_targeted_frames,
    get_work_dir,
    probe_duration,
)
from utils.logging_setup import configure_logging, get_console
from utils.models import (
    ClassificationOutput,
    ClipClassification,
    ClipManifest,
    DetectionsRefined,
    RefinedDetection,
)
from utils.ollama_client import OllamaClient, resolve_host
from utils.prompt_templates import (
    BUILTIN_DOMAINS,
    build_classification_prompt,
    get_template_for_keywords,
    load_custom_templates,
    resolve_domain_from_keywords,
    resolve_template_for_subject,
)
from utils.run_archive import archive_run_artifacts, get_run_dir
from utils.signals import setup_signal_handlers
from utils.stats import (
    Phase0Stats,
    Phase1Stats,
    Phase2Stats,
    PipelineStats,
    format_stats_log,
    format_stats_table,
    parse_duration,
    save_stats_json,
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


def get_clip_id(clip: dict[str, Any] | ClipManifest) -> str:
    """Derive a stable clip identifier matching manifest and intra-clip checkpoints."""
    if isinstance(clip, ClipManifest):
        raw_clip_id = clip.clip_id
        file_path = clip.file_path
    else:
        raw_clip_id = clip.get("clip_id")
        file_path = clip.get("file_path", "")

    if not raw_clip_id:
        if file_path:
            raw_clip_id = f"clip_{hashlib.sha256(str(file_path).encode('utf-8')).hexdigest()[:16]}"
        else:
            raw_clip_id = "clip"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw_clip_id))[:64]


def select_representative_frames(frames: list[dict], count: int) -> list[dict]:
    """Select at most count evenly-spaced representative frames from extracted frames."""
    if count <= 0:
        return []
    if len(frames) <= count:
        return list(frames)
    step = len(frames) / (count + 1)
    indices = [int((i + 1) * step) for i in range(count)]
    return [frames[idx] for idx in indices]


def extract_classification_frames(
    file_path: str | Path,
    output_dir: Path,
    count: int = 3,
    scale: str | None = None,
    fps: float | None = None,
    duration: float | None = None,
    interval_seconds: float = 300.0,
    max_count: int = 8,
) -> list[dict]:
    """Extract representative frames for domain classification using fast targeted extraction."""
    if count <= 0:
        return []

    dur = duration if duration is not None and duration > 0 else probe_duration(file_path)
    if dur is not None and dur > 0:
        timestamps = calculate_sample_timestamps(
            duration=dur,
            min_count=count,
            interval_seconds=interval_seconds,
            max_count=max_count,
        )
        targeted = extract_targeted_frames(
            video_path=file_path,
            output_dir=output_dir,
            timestamps=timestamps,
            scale=scale,
        )
        if targeted:
            return targeted

    # Fallback to extract_frames if duration probe or targeted seek fails
    actual_fps = fps or 0.25
    frames = extract_frames(
        video_path=file_path,
        output_dir=output_dir,
        fps=actual_fps,
        scale=scale,
    )
    return select_representative_frames(frames, count)


def aggregate_frame_classifications(
    frame_results: list[dict[str, Any]],
) -> tuple[str, float | None, bool, str | None]:
    """Aggregate frame-level classification results into winning domain, confidence, and uncertainty.

    Returns:
        (winning_domain, mean_confidence, is_uncertain, reason)
    """
    if not frame_results:
        return "generic", None, True, "no usable responses"

    # Check usability flags and quality scores across sampled frames
    usable_flags = [bool(item["usable"]) for item in frame_results if "usable" in item]
    quality_scores = [
        int(item["quality_score"])
        for item in frame_results
        if item.get("quality_score") is not None
    ]

    is_unusable = False
    if usable_flags and len(usable_flags) >= 1:
        unusable_ratio = (len(usable_flags) - sum(usable_flags)) / len(usable_flags)
        if unusable_ratio >= 0.66 or (
            quality_scores and (sum(quality_scores) / len(quality_scores)) <= 2.5
        ):
            is_unusable = True

    if is_unusable:
        return (
            "unusable",
            1.0,
            False,
            "unusable camera footage (inside bag, covered lens, or dangling)",
        )

    domain_counts: dict[str, int] = {}
    domain_confidences: dict[str, list[float]] = {}

    for item in frame_results:
        dom = str(item.get("domain", "")).strip().lower()
        if not dom:
            dom = "generic"
        try:
            conf = float(item["confidence"])
        except KeyError, ValueError, TypeError:
            conf = 0.5
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        domain_confidences.setdefault(dom, []).append(conf)

    unique_domains = list(domain_counts.keys())
    winning_domain = max(
        unique_domains,
        key=lambda d: (domain_counts[d], sum(domain_confidences[d])),
    )
    winning_confs = domain_confidences[winning_domain]
    mean_confidence = round(sum(winning_confs) / len(winning_confs), 3)

    has_disagreement = len(unique_domains) > 1
    has_low_confidence = mean_confidence < 0.6

    if has_disagreement and has_low_confidence:
        return (
            winning_domain,
            mean_confidence,
            True,
            f"sampled frames disagree on domain and low confidence ({mean_confidence:.2f} < 0.6)",
        )
    if has_disagreement:
        return winning_domain, mean_confidence, True, "sampled frames disagree on domain"
    if has_low_confidence:
        return (
            winning_domain,
            mean_confidence,
            True,
            f"low confidence ({mean_confidence:.2f} < 0.6)",
        )

    return winning_domain, mean_confidence, False, None


def _clean_env_string(raw_env: Any) -> str:
    """Normalize environment string from scalar or list representation."""
    if isinstance(raw_env, str):
        return raw_env.strip()
    if isinstance(raw_env, list):
        return ", ".join(str(item).strip() for item in raw_env if str(item).strip())
    if raw_env is not None:
        return str(raw_env).strip()
    return ""


def _clean_text_list(raw_values: Any) -> list[str]:
    """Extract and deduplicate cleaned alphanumeric strings from list or comma-separated string."""
    items: list[str] = []
    if isinstance(raw_values, list):
        candidates = raw_values
    elif isinstance(raw_values, str) and raw_values.strip():
        candidates = raw_values.split(",")
    else:
        return []

    for item in candidates:
        s = str(item).strip().rstrip(".,;")
        if s and s not in items:
            items.append(s)
    return items


def aggregate_frame_context(
    frame_results: list[dict[str, Any]],
    winning_domain: str | None = None,
) -> tuple[str, list[str]]:
    """Aggregate inferred environment and primary subjects across sample frames.

    Prioritizes usable frames matching the winning domain to avoid contamination from
    discordant or unusable frames. Returns (environment, primary_subjects).
    """
    if not frame_results or winning_domain == "unusable":
        return "", []

    # Filter to usable frames
    usable_frames = [f for f in frame_results if f.get("usable", True)]
    if not usable_frames:
        usable_frames = frame_results

    # Prioritize frames matching the winning domain if available
    domain_frames = (
        [f for f in usable_frames if f.get("domain") == winning_domain] if winning_domain else []
    )
    candidate_frames = domain_frames if domain_frames else usable_frames

    # Aggregate environment
    env_counts: dict[str, int] = {}
    env_display: dict[str, str] = {}
    env_max_conf: dict[str, float] = {}

    for frame in candidate_frames:
        env = _clean_env_string(frame.get("environment"))
        if not env:
            continue
        key = env.lower()
        env_counts[key] = env_counts.get(key, 0) + 1
        if key not in env_display:
            env_display[key] = env
            env_max_conf[key] = float(frame.get("confidence") or 0.0)
        else:
            env_max_conf[key] = max(env_max_conf[key], float(frame.get("confidence") or 0.0))

    aggregated_env = ""
    if env_counts:
        # Sort by frequency (descending), then confidence (descending)
        best_env_key = max(
            env_counts.keys(),
            key=lambda k: (env_counts[k], env_max_conf.get(k, 0.0)),
        )
        aggregated_env = env_display[best_env_key]

    # Aggregate primary subjects (ranked by cross-frame frequency)
    subject_frame_counts: dict[str, int] = {}
    subject_display: dict[str, str] = {}
    first_seen_order: list[str] = []

    for frame in candidate_frames:
        seen_in_this_frame: set[str] = set()
        for s_clean in _clean_text_list(frame.get("primary_subjects", [])):
            key = s_clean.lower()
            if key not in seen_in_this_frame:
                seen_in_this_frame.add(key)
                subject_frame_counts[key] = subject_frame_counts.get(key, 0) + 1
                if key not in subject_display:
                    subject_display[key] = s_clean
                    first_seen_order.append(key)

    # Sort unique subjects by frame frequency (descending), preserving first-seen order on ties
    sorted_subject_keys = sorted(
        first_seen_order,
        key=lambda k: subject_frame_counts[k],
        reverse=True,
    )
    aggregated_subjects = [subject_display[k] for k in sorted_subject_keys]

    return aggregated_env, aggregated_subjects


def classify_clip(
    clip: dict[str, Any] | ClipManifest,
    sample_frames: list[dict[str, Any]],
    client: OllamaClient,
    ollama_host: str,
    ollama_model: str,
    ollama_timeout: int,
    registry: dict[str, Any],
    phase0_stats: Phase0Stats | None = None,
) -> ClipClassification:
    """Classify video domain across sample frames, returning a validated ClipClassification."""
    file_name = clip.file_name if isinstance(clip, ClipManifest) else clip.get("file_name", "")
    file_path = clip.file_path if isinstance(clip, ClipManifest) else clip.get("file_path", "")
    clip_id = get_clip_id(clip)

    if not sample_frames:
        return ClipClassification(
            clip_id=clip_id,
            file_name=file_name,
            file_path=str(file_path),
            domain="generic",
            confidence=None,
            uncertain=True,
            reason="no usable responses",
            environment="",
            primary_subjects=[],
        )

    prompt = build_classification_prompt(registry)
    usable_results: list[dict[str, Any]] = []

    for frame in sample_frames:
        start_time = time.perf_counter()
        result = call_ollama(
            ollama_host,
            ollama_model,
            frame["frame_path"],
            prompt,
            ollama_timeout,
            client=client,
            phase="classify",
        )
        elapsed = time.perf_counter() - start_time
        if phase0_stats is not None:
            phase0_stats.frames_analyzed += 1
            phase0_stats.elapsed_seconds += elapsed
            phase0_stats.ollama_calls += 1

        if not result or not isinstance(result, dict):
            if phase0_stats is not None:
                phase0_stats.ollama_failures += 1
            continue

        raw_domain = str(result.get("domain", "")).strip().lower()
        conf_raw = result.get("confidence")
        if not raw_domain or conf_raw is None:
            if phase0_stats is not None:
                phase0_stats.ollama_failures += 1
            continue

        try:
            confidence = float(conf_raw)
        except ValueError, TypeError:
            if phase0_stats is not None:
                phase0_stats.ollama_failures += 1
            continue

        matched_domain = None
        for dom in registry:
            if dom in raw_domain or raw_domain in dom:
                matched_domain = dom
                break
        if not matched_domain:
            matched_domain = "generic"

        usable_val = bool(result.get("usable", True))
        try:
            q_score = (
                int(result["quality_score"]) if result.get("quality_score") is not None else None
            )
        except ValueError, TypeError:
            q_score = None

        usable_results.append({
            "domain": matched_domain,
            "confidence": max(0.0, min(1.0, confidence)),
            "usable": usable_val,
            "quality_score": q_score,
            "environment": _clean_env_string(result.get("environment")),
            "primary_subjects": _clean_text_list(result.get("primary_subjects")),
        })

    winning_domain, mean_conf, is_uncertain, reason = aggregate_frame_classifications(
        usable_results
    )

    usable_clip = winning_domain != "unusable"

    aggregated_env, aggregated_subjects = aggregate_frame_context(
        usable_results, winning_domain=winning_domain
    )

    return ClipClassification(
        clip_id=clip_id,
        file_name=file_name,
        file_path=str(file_path),
        domain=winning_domain,
        confidence=mean_conf,
        uncertain=is_uncertain,
        reason=reason,
        usable=usable_clip,
        environment=aggregated_env,
        primary_subjects=aggregated_subjects,
    )


def classify_clip_domain(
    clip: dict,
    sample_frames: list[dict],
    client: OllamaClient,
    ollama_host: str,
    ollama_model: str,
    ollama_timeout: int,
    registry: dict[str, Any],
    phase0_stats: Phase0Stats | None = None,
) -> str:
    """Classify video domain by evaluating sample frames with the VLM."""
    classification = classify_clip(
        clip=clip,
        sample_frames=sample_frames,
        client=client,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        ollama_timeout=ollama_timeout,
        registry=registry,
        phase0_stats=phase0_stats,
    )
    return classification.domain


def load_and_validate_classifications(
    classifications_path: Path | str,
) -> list[ClipClassification]:
    """Load and validate classifications from a JSON file, returning ClipClassification records."""
    path = Path(classifications_path)
    if not path.exists():
        raise FileNotFoundError(f"Classifications file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "classifications" in data:
            output = ClassificationOutput.model_validate(data)
            return output.classifications
        return [ClipClassification.model_validate(data)]
    if isinstance(data, list):
        return [ClipClassification.model_validate(item) for item in data]

    raise ValueError(f"Invalid classification format in {path}: expected dict or list")


def merge_classifications_into_manifest(
    clips: list[dict[str, Any]],
    classifications: list[ClipClassification],
) -> tuple[int, int, list[str]]:
    """Merge confident classifications into in-memory clips; ignore uncertain ones.

    Returns:
        (reused_count, ignored_uncertain_count, unmatched_clip_ids)
    """
    class_map = {c.clip_id: c for c in classifications}
    reused_count = 0
    ignored_uncertain_count = 0
    manifest_clip_ids = set()

    for clip in clips:
        cid = get_clip_id(clip)
        manifest_clip_ids.add(cid)
        if cid in class_map:
            record = class_map[cid]
            if not record.uncertain and record.domain:
                clip["domain"] = record.domain
                reused_count += 1
                logger.info(
                    "[Pass 0] Reused confident domain '%s' (conf=%.2f) for %s from classifications",
                    record.domain,
                    record.confidence or 0.0,
                    clip.get("file_name", cid),
                )
            else:
                ignored_uncertain_count += 1
                logger.info(
                    "[Pass 0] Ignored uncertain classification '%s' for %s (reason: %s); will classify in Pass 0",
                    record.domain,
                    clip.get("file_name", cid),
                    record.reason or "uncertain",
                )

    unmatched_ids = [c.clip_id for c in classifications if c.clip_id not in manifest_clip_ids]
    if unmatched_ids:
        logger.warning(
            "Found %d classification record(s) not matching any input clip: %s",
            len(unmatched_ids),
            ", ".join(unmatched_ids[:5]),
        )

    return reused_count, ignored_uncertain_count, unmatched_ids


def save_classifications_atomic(
    output_path: Path | str,
    output_obj: ClassificationOutput,
) -> None:
    """Atomically write classification output to disk using a temporary file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    try:
        temp_path.write_text(output_obj.model_dump_json(indent=2), encoding="utf-8")
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def classify_footage(
    clips: list[dict[str, Any]],
    config: dict[str, Any],
    work_dir: Path | None = None,
    client: OllamaClient | None = None,
    output_file: Path | str | list[Path | str] | None = None,
    resume: bool = False,
) -> list[ClipClassification]:
    """Run Pass 0 standalone domain classification across all clips in a folder/manifest."""
    if not clips:
        return []

    output_paths: list[Path] = []
    if output_file is not None:
        if isinstance(output_file, (list, tuple)):
            output_paths = [Path(p) for p in output_file]
        else:
            output_paths = [Path(output_file)]

    existing_map: dict[str, ClipClassification] = {}
    if resume:
        for p in output_paths:
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    loaded = ClassificationOutput.model_validate(data)
                    for item in loaded.classifications:
                        if item.clip_id:
                            existing_map[item.clip_id] = item
                        if item.file_name:
                            existing_map[item.file_name] = item
                    logger.info(
                        "Loaded %d existing classification(s) for resume from %s",
                        len(loaded.classifications),
                        p,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Could not load classifications for resume from %s: %s",
                        p,
                        exc,
                    )

    scan_params = config.get("scan_parameters", {})
    processing_cfg = config.get("processing", {})
    use_ramdisk = bool(
        scan_params.get(
            "use_ramdisk",
            processing_cfg.get("use_ramdisk", True),
        )
    )
    temp_work_dir = False
    if work_dir is None:
        work_dir = get_work_dir(use_ramdisk=use_ramdisk)
        temp_work_dir = True

    ollama_cfg = config.get("ollama", {})
    ollama_host = resolve_host(ollama_cfg.get("host"))
    ollama_model = ollama_cfg.get("model", "qwen2.5-vl:7b")
    ollama_timeout = ollama_cfg.get("timeout_seconds", 90)
    ollama_client = client or OllamaClient(ollama_host, ollama_model, ollama_timeout)

    registry = load_custom_templates(config)
    classify_frames_count = scan_params.get("classify_frames_count", 3)
    coarse_scale = scan_params.get("coarse_scale", "426:-2")

    results: list[ClipClassification] = []

    try:
        for idx, clip in enumerate(clips, start=1):
            file_path = clip.get("file_path", "")
            clip_name = clip.get("file_name", Path(file_path).name if file_path else f"clip_{idx}")
            clip_id = get_clip_id(clip)

            if resume and (clip_id in existing_map or clip_name in existing_map):
                classification = existing_map.get(clip_id) or existing_map[clip_name]
                logger.info(
                    "Resuming %s (%d/%d): using existing classification (%s)",
                    clip_name,
                    idx,
                    len(clips),
                    classification.domain,
                )
                results.append(classification)
                if output_paths:
                    out_obj = ClassificationOutput(
                        total_clips=len(results),
                        classifications=results,
                    )
                    for p in output_paths:
                        try:
                            save_classifications_atomic(p, out_obj)
                        except Exception as exc:
                            logger.warning(
                                "Failed to incrementally save classifications to %s: %s",
                                p,
                                exc,
                            )
                continue

            clip_work = work_dir / f"classify_{clip_id}"
            clip_work.mkdir(parents=True, exist_ok=True)

            logger.info("Classifying clip %d/%d: %s", idx, len(clips), clip_name)
            try:
                sample_frames = extract_classification_frames(
                    file_path=file_path,
                    output_dir=clip_work,
                    count=classify_frames_count,
                    scale=coarse_scale,
                )
                classification = classify_clip(
                    clip=clip,
                    sample_frames=sample_frames,
                    client=ollama_client,
                    ollama_host=ollama_host,
                    ollama_model=ollama_model,
                    ollama_timeout=ollama_timeout,
                    registry=registry,
                )
            except Exception as exc:
                logger.warning("Error classifying %s: %s", clip_name, exc)
                classification = ClipClassification(
                    clip_id=clip_id,
                    file_name=clip_name,
                    file_path=str(file_path),
                    domain="generic",
                    confidence=None,
                    uncertain=True,
                    reason=f"error: {exc}",
                    environment="",
                    primary_subjects=[],
                )
            results.append(classification)
            cleanup_work_dir(clip_work)

            if output_paths:
                out_obj = ClassificationOutput(
                    total_clips=len(results),
                    classifications=results,
                )
                for p in output_paths:
                    try:
                        save_classifications_atomic(p, out_obj)
                    except Exception as exc:
                        logger.warning(
                            "Failed to incrementally save classifications to %s: %s",
                            p,
                            exc,
                        )
    finally:
        if temp_work_dir:
            cleanup_work_dir(work_dir)

    return results


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
    checkpoints_dir: Path | None = None,
    resume: bool = False,
    phase0_stats: Phase0Stats | None = None,
    prefetched_coarse_frames: list[dict] | None = None,
    on_coarse_start: Callable[[], None] | None = None,
) -> list[RefinedDetection]:
    """Process a single clip through 3-stage domain-aware scanning.

    Pass 0: Video domain classification (sample frames).
    Pass 1: Domain coarse scan (candidate windows & subject identification).
    Pass 2: Subject-routed fine scan (quality scoring & refined highlight windows).

    Returns a list of refined detections (may be empty).
    """
    file_path = clip["file_path"]
    if not Path(file_path).exists():
        logger.warning("File not found: %s", file_path)
        return []

    clip_id = get_clip_id(clip)
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
    checkpoint_interval = scan_params.get("checkpoint_interval_frames", 20)

    checkpoint: ClipScanCheckpoint | None = None
    checkpoint_file: Path | None = None
    if checkpoints_dir is not None:
        checkpoint_file = checkpoints_dir / f"{clip_id}.json"
        if resume and checkpoint_file.exists():
            checkpoint = ClipScanCheckpoint.load(checkpoint_file)
            if checkpoint is not None:
                if checkpoint.pass_stage == "completed":
                    if checkpoint.detected_domain == "unusable":
                        logger.info(
                            "Skipping %s (marked unusable in checkpoint)", clip["file_name"]
                        )
                        cleanup_work_dir(clip_work_dir)
                        return []
                    logger.info("Skipping %s (already completed in checkpoint)", clip["file_name"])
                    cleanup_work_dir(clip_work_dir)
                    return checkpoint.refined_detections
                logger.info(
                    "Resumed %s from intra-clip checkpoint (stage=%s, domain=%s, %d coarse frames evaluated)",
                    clip["file_name"],
                    checkpoint.pass_stage,
                    checkpoint.detected_domain or "unknown",
                    len(checkpoint.coarse_results),
                )
        if checkpoint is None:
            checkpoint = ClipScanCheckpoint(
                clip_id=clip_id,
                file_name=clip["file_name"],
                coarse_fps=coarse_fps,
            )

    ollama_cfg = config.get("ollama", {})
    ollama_host = resolve_host(ollama_cfg.get("host"))
    ollama_model = ollama_cfg.get("model", "qwen2.5-vl:7b")
    ollama_timeout = ollama_cfg.get("timeout_seconds", 90)
    client = ollama_client or OllamaClient(ollama_host, ollama_model, ollama_timeout)

    tags = clip.get("tags", [])

    # ---------- Pass 0: Domain Resolution / Classification ----------
    registry = load_custom_templates(config)
    configured_domain = scan_params.get("domain")
    classify_clips = scan_params.get("classify_clips")
    if classify_clips is None:
        classify_clips = "auto" if configured_domain == "auto" else "off"
    classify_frames_count = max(1, int(scan_params.get("classify_frames_count", 3)))
    classify_interval_seconds = float(scan_params.get("classify_interval_seconds", 300.0))
    classify_frames_max = int(scan_params.get("classify_frames_max", 8))

    detected_domain: str | None = None
    if checkpoint is not None and checkpoint.detected_domain:
        detected_domain = checkpoint.detected_domain
        logger.info(
            "[Pass 0] Resumed detected domain '%s' for %s from checkpoint",
            detected_domain,
            clip["file_name"],
        )
    elif configured_domain and configured_domain != "auto" and classify_clips != "always":
        detected_domain = configured_domain
        logger.info(
            "[Pass 0] Using configured domain '%s' for %s",
            detected_domain,
            clip["file_name"],
        )
    elif clip.get("domain") and classify_clips != "always":
        detected_domain = clip["domain"]
        logger.info(
            "[Pass 0] Using clip manifest domain '%s' for %s",
            detected_domain,
            clip["file_name"],
        )
    elif classify_clips != "always" and tags:
        inferred = resolve_domain_from_keywords(tags, registry)
        if inferred:
            detected_domain = inferred
            logger.info(
                "[Pass 0] Inferred domain '%s' from clip tags for %s",
                detected_domain,
                clip["file_name"],
            )

    # Execute Pass 0 VLM classification if domain is still unknown or forced via 'always'
    need_classification = (classify_clips == "always") or (
        detected_domain is None and classify_clips != "off"
    )
    if need_classification:
        logger.info("[Pass 0] Extracting frames @ %.1f FPS: %s", coarse_fps, clip["file_name"])
        if state is not None:
            state.set_activity(f"scan: {clip['file_name']} (pass 0 extract, {coarse_fps} FPS)")

        pass0_dir = clip_work_dir / "pass0"
        sample_frames = extract_classification_frames(
            file_path=file_path,
            output_dir=pass0_dir,
            count=classify_frames_count,
            scale=coarse_scale,
            fps=coarse_fps,
            duration=clip.get("duration"),
            interval_seconds=classify_interval_seconds,
            max_count=classify_frames_max,
        )

        logger.info(
            "[Pass 0] Extracted %d frames from %s",
            len(sample_frames),
            clip["file_name"],
        )

        if sample_frames:
            logger.info(
                "[Pass 0] Classifying domain for %s across %d sample frames...",
                clip["file_name"],
                len(sample_frames),
            )
            detected_domain = classify_clip_domain(
                clip,
                sample_frames,
                client,
                ollama_host,
                ollama_model,
                ollama_timeout,
                registry,
                phase0_stats=phase0_stats,
            )
            if phase0_stats is not None:
                phase0_stats.clips_classified += 1
                phase0_stats.domain_counts[detected_domain] = (
                    phase0_stats.domain_counts.get(detected_domain, 0) + 1
                )
            logger.info(
                "[Pass 0] Classified %s as domain '%s'",
                clip["file_name"],
                detected_domain,
            )
        cleanup_work_dir(pass0_dir)

    # Check for unusable or non-target footage and early-exit to avoid full decode
    if detected_domain == "unusable":
        msg = f"⚡ Skipped {clip['file_name']}: unusable camera footage (inside bag / lens covered / dangling)"
        logger.warning(msg)
        get_console().print(f"[yellow]{msg}[/yellow]")
        if phase0_stats is not None:
            phase0_stats.clips_unusable += 1
        if checkpoint is not None:
            checkpoint.detected_domain = "unusable"
            checkpoint.pass_stage = "completed"
            if checkpoint_file is not None:
                checkpoint.save_atomic(checkpoint_file)
        cleanup_work_dir(clip_work_dir)
        return []

    if configured_domain and configured_domain != "auto" and detected_domain != configured_domain:
        msg = f"⚡ Skipped {clip['file_name']}: non-target domain '{detected_domain}' (configured: '{configured_domain}')"
        logger.warning(msg)
        get_console().print(f"[yellow]{msg}[/yellow]")
        if phase0_stats is not None:
            phase0_stats.clips_skipped_domain += 1
        if checkpoint is not None:
            checkpoint.detected_domain = detected_domain
            checkpoint.pass_stage = "completed"
            if checkpoint_file is not None:
                checkpoint.save_atomic(checkpoint_file)
        cleanup_work_dir(clip_work_dir)
        return []

    if detected_domain is None:
        detected_domain = "underwater"

    if checkpoint is not None:
        checkpoint.detected_domain = detected_domain

    # ---------- Pass 1: Coarse scan setup ----------
    candidate_windows: list[tuple[float, float]] = []
    coarse_hits: list[tuple[float, tuple[float, float], str, str, float]] = []
    coarse_confidences: list[float] = []
    coarse_detected = 0
    coarse_failures = 0
    coarse_elapsed_total = 0.0

    skip_coarse_inference = checkpoint is not None and checkpoint.pass_stage in (
        "coarse_done",
        "fine",
        "completed",
    )

    coarse_frames: list[dict] = []
    if skip_coarse_inference and checkpoint is not None:
        logger.info(
            "[Pass 1] Skipping coarse scan for %s (already completed in checkpoint)",
            clip["file_name"],
        )
        candidate_windows = list(checkpoint.candidate_windows)
        coarse_confidences = list(checkpoint.coarse_confidences)
        coarse_detected = checkpoint.coarse_detected
        coarse_failures = checkpoint.coarse_failures
        coarse_elapsed_total = checkpoint.coarse_elapsed_total
        if phase1_stats is not None:
            phase1_stats.frames_analyzed += checkpoint.coarse_frames_analyzed
            phase1_stats.detections_detected += coarse_detected
        coarse_frames_count = checkpoint.coarse_frames_analyzed
    else:
        logger.info(
            "[Pass 1] Coarse scan @ %.1f FPS (%s): %s",
            coarse_fps,
            detected_domain,
            clip["file_name"],
        )
        if state is not None:
            state.set_activity(f"scan: {clip['file_name']} (pass 1 coarse, {coarse_fps} FPS)")

        if prefetched_coarse_frames:
            coarse_frames = prefetched_coarse_frames
            logger.info(
                "[Pass 1] Using %d prefetched coarse frames for %s",
                len(coarse_frames),
                clip["file_name"],
            )
        else:
            coarse_frames = extract_frames(
                video_path=file_path,
                output_dir=clip_work_dir / "coarse",
                fps=coarse_fps,
                scale=coarse_scale,
            )
        coarse_frames_count = len(coarse_frames)
        if phase1_stats is not None:
            phase1_stats.frames_extracted += len(coarse_frames)
        logger.info(
            "[Pass 1] Extracted %d frames from %s",
            len(coarse_frames),
            clip["file_name"],
        )

        if on_coarse_start is not None:
            try:
                on_coarse_start()
            except Exception as exc:
                logger.warning("on_coarse_start callback failed: %s", exc)

    dom_config = registry.get(
        detected_domain, registry.get("underwater", BUILTIN_DOMAINS["underwater"])
    )

    matched_template = get_template_for_keywords(tags, dom_config.templates) if tags else None
    coarse_prompt = matched_template.coarse_prompt if matched_template else dom_config.coarse_prompt

    if not skip_coarse_inference:
        task_id = None
        if progress is not None and coarse_frames:
            task_id = progress.add_task(
                f"[cyan]Pass 1 ({detected_domain}):[/] {clip['file_name']}",
                total=len(coarse_frames),
                status="(0 hits)",
            )

        try:
            for frame_index, frame in enumerate(coarse_frames, start=1):
                ts = frame["timestamp"]
                frame_key = f"{round(ts, 3):.3f}"
                if checkpoint is not None and frame_key in checkpoint.coarse_results:
                    cached = checkpoint.coarse_results[frame_key]
                    if phase1_stats is not None:
                        phase1_stats.frames_analyzed += 1
                    coarse_confidences.append(cached.confidence)
                    if cached.detected:
                        coarse_detected += 1
                        if phase1_stats is not None:
                            phase1_stats.detections_detected += 1
                    if cached.candidate_window is not None:
                        candidate_windows.append(cached.candidate_window)
                        coarse_hits.append((
                            ts,
                            cached.candidate_window,
                            cached.subject,
                            cached.category,
                            cached.confidence,
                        ))
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)
                        progress.update(task_id, status=f"({len(candidate_windows)} hits)")
                    continue

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
                subject = str(result.get("subject", "") or "")
                category = str(result.get("category", "") or "")
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
                    cand_win = (max(0, ts - fine_padding), ts + fine_padding)
                    candidate_windows.append(cand_win)
                    coarse_hits.append((ts, cand_win, subject, category, confidence))
                else:
                    cand_win = None

                if progress is not None and task_id is not None:
                    progress.advance(task_id)
                    progress.update(task_id, status=f"({len(candidate_windows)} hits)")

                logger.debug(
                    "[Pass 1] Frame %d @ %.1fs: detected=%s confidence=%.2f quality=%d passes_threshold=%s "
                    "subject=%s category=%s (%.1fs)",
                    frame_index,
                    frame["timestamp"],
                    detected,
                    confidence,
                    quality_score,
                    frame_passes,
                    subject or "-",
                    category or "-",
                    elapsed,
                )

                if checkpoint is not None:
                    checkpoint.coarse_results[frame_key] = CoarseFrameResult(
                        timestamp=ts,
                        detected=detected,
                        confidence=confidence,
                        quality_score=quality_score,
                        subject=subject,
                        category=category,
                        candidate_window=cand_win,
                        elapsed=elapsed,
                    )
                    checkpoint.coarse_frames_analyzed = len(checkpoint.coarse_results)
                    checkpoint.candidate_windows = candidate_windows
                    checkpoint.coarse_confidences = coarse_confidences
                    checkpoint.coarse_detected = coarse_detected
                    checkpoint.coarse_failures = coarse_failures
                    checkpoint.coarse_elapsed_total = coarse_elapsed_total

                    if (
                        len(checkpoint.coarse_results) % checkpoint_interval == 0
                        and checkpoint_file is not None
                    ):
                        checkpoint.save_atomic(checkpoint_file)
        except KeyboardInterrupt:
            logger.warning(
                "\nScan interrupted during Pass 1 for %s. Saving intra-clip checkpoint...",
                clip["file_name"],
            )
            if checkpoint is not None and checkpoint_file is not None:
                checkpoint.candidate_windows = candidate_windows
                checkpoint.coarse_confidences = coarse_confidences
                checkpoint.coarse_detected = coarse_detected
                checkpoint.coarse_failures = coarse_failures
                checkpoint.coarse_elapsed_total = coarse_elapsed_total
                checkpoint.save_atomic(checkpoint_file)
            raise
        finally:
            if progress is not None and task_id is not None:
                progress.remove_task(task_id)

        if checkpoint is not None:
            checkpoint.pass_stage = "coarse_done"
            checkpoint.candidate_windows = candidate_windows
            checkpoint.coarse_confidences = coarse_confidences
            checkpoint.coarse_detected = coarse_detected
            checkpoint.coarse_failures = coarse_failures
            checkpoint.coarse_elapsed_total = coarse_elapsed_total
            if checkpoint_file is not None:
                checkpoint.save_atomic(checkpoint_file)

    logger.info(
        "[Pass 1] Summary %s: %d frames, %d detected, %d over threshold %.2f, "
        "confidence min/avg/max = %s, %d failed calls, %.1fs total",
        clip["file_name"],
        coarse_frames_count,
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

    # Reconstruct coarse_hits from checkpoint if resuming
    if not coarse_hits and checkpoint is not None:
        coarse_hits.extend(
            (r.timestamp, r.candidate_window, r.subject, r.category, r.confidence)
            for r in checkpoint.coarse_results.values()
            if r.detected and r.candidate_window is not None
        )

    window_routing: dict[str, dict[str, str]] = (
        dict(checkpoint.window_routing)
        if checkpoint is not None and checkpoint.window_routing
        else {}
    )
    for window_index, (win_start, win_end) in enumerate(merged, start=1):
        win_key = str(window_index)
        if win_key not in window_routing:
            matching_hits = [h for h in coarse_hits if (win_start - 0.5 <= h[0] <= win_end + 0.5)]
            if matching_hits:
                best_hit = max(matching_hits, key=lambda h: h[4])
                window_routing[win_key] = {"subject": best_hit[2], "category": best_hit[3]}
            else:
                window_routing[win_key] = {"subject": "", "category": ""}

    if checkpoint is not None:
        checkpoint.window_routing = window_routing

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

    if checkpoint is not None:
        checkpoint.pass_stage = "fine"
        all_detections = list(checkpoint.refined_detections)
        fine_frames_analyzed = checkpoint.fine_frames_analyzed
        fine_passed = checkpoint.fine_passed
        fine_failures = checkpoint.fine_failures
        fine_elapsed_total = checkpoint.fine_elapsed_total
        fine_confidences = list(checkpoint.fine_confidences)

    try:
        for window_index, (win_start, win_end) in enumerate(merged, start=1):
            if checkpoint is not None and window_index in checkpoint.fine_windows_completed:
                logger.info(
                    "[Pass 2] Skipping window %d of %d (already completed in checkpoint)",
                    window_index,
                    len(merged),
                )
                continue

            win_route = window_routing.get(str(window_index), {})
            win_subject = win_route.get("subject", "")
            win_category = win_route.get("category", "")
            window_template = resolve_template_for_subject(
                subject=win_subject,
                category=win_category,
                domain=detected_domain,
                tags=tags,
                registry=registry,
            )
            fine_prompt = window_template.fine_prompt
            color = window_template.color

            duration = win_end - win_start
            win_dir = clip_work_dir / "fine" / f"win_{int(win_start)}_{int(win_end)}"
            logger.info(
                "[Pass 2] Window %d of %d: %.1fs-%.1fs (%.1fs) @ %.1f FPS [template: %s, domain: %s]",
                window_index,
                len(merged),
                win_start,
                win_end,
                duration,
                fine_fps,
                window_template.name,
                detected_domain,
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
                                subject=result.get("subject", win_subject or "unknown"),
                                confidence=confidence,
                                quality_score=quality_score,
                                description=result.get("description", ""),
                                color=color,
                                template=window_template.name,
                                domain=detected_domain,
                            )
                        )
                    except (ValueError, TypeError) as exc:
                        logger.warning("Skipping invalid detection: %s", exc)
                        continue
            finally:
                if progress is not None and fine_task_id is not None:
                    progress.remove_task(fine_task_id)

            if checkpoint is not None:
                checkpoint.fine_windows_completed.append(window_index)
                checkpoint.refined_detections = list(all_detections)
                checkpoint.fine_frames_analyzed = fine_frames_analyzed
                checkpoint.fine_passed = fine_passed
                checkpoint.fine_failures = fine_failures
                checkpoint.fine_elapsed_total = fine_elapsed_total
                checkpoint.fine_confidences = fine_confidences
                if checkpoint_file is not None:
                    checkpoint.save_atomic(checkpoint_file)
    except KeyboardInterrupt:
        logger.warning(
            "\nScan interrupted during Pass 2 for %s. Saving intra-clip checkpoint...",
            clip["file_name"],
        )
        if checkpoint is not None and checkpoint_file is not None:
            checkpoint.refined_detections = list(all_detections)
            checkpoint.fine_frames_analyzed = fine_frames_analyzed
            checkpoint.fine_passed = fine_passed
            checkpoint.fine_failures = fine_failures
            checkpoint.fine_elapsed_total = fine_elapsed_total
            checkpoint.fine_confidences = fine_confidences
            checkpoint.save_atomic(checkpoint_file)
        raise

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

    if checkpoint is not None:
        checkpoint.pass_stage = "completed"
        checkpoint.refined_detections = list(all_detections)
        if checkpoint_file is not None:
            checkpoint.save_atomic(checkpoint_file)

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
    parser.add_argument(
        "--classifications",
        type=Path,
        default=None,
        help="Path to pre-computed classifications JSON to reuse confident domain labels",
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

    checkpoints_dir = (target_run_dir or output_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

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
    if args.classifications is not None:
        try:
            loaded_classifications = load_and_validate_classifications(args.classifications)
            reused_cnt, ignored_cnt, unmatched = merge_classifications_into_manifest(
                clips, loaded_classifications
            )
            logger.info(
                "Classifications processed: %d confident reused, %d uncertain ignored, %d unmatched",
                reused_cnt,
                ignored_cnt,
                len(unmatched),
            )
        except Exception as exc:
            logger.error("Failed to load classifications from %s: %s", args.classifications, exc)
            sys.exit(1)

    if args.resume and state is not None:
        for clip in clips:
            cname = clip.get("file_name", "unknown")
            if state.is_item_done("scan", cname):
                completed_clips.add(cname)

    scan_start_time = time.perf_counter()
    p0_stats = Phase0Stats()
    p1_stats = Phase1Stats()
    p2_stats = Phase2Stats()
    if args.resume and state is not None:
        prev_scan_stats = state.get("scan_stats")
        if isinstance(prev_scan_stats, dict):
            try:
                prev_p0 = prev_scan_stats.get("phase0")
                prev_p1 = prev_scan_stats.get("phase1")
                prev_p2 = prev_scan_stats.get("phase2")
                if isinstance(prev_p0, dict):
                    p0_stats = Phase0Stats.model_validate(prev_p0)
                if isinstance(prev_p1, dict):
                    p1_stats = Phase1Stats.model_validate(prev_p1)
                if isinstance(prev_p2, dict):
                    p2_stats = Phase2Stats.model_validate(prev_p2)
            except Exception as exc:
                logger.debug("Could not restore previous scan_stats: %s", exc)

    scan_params = config.get("scan_parameters", {})
    prefetch_enabled = bool(scan_params.get("prefetch_coarse", True))
    coarse_fps = scan_params.get("coarse_fps", 0.5)
    coarse_scale = scan_params.get("coarse_scale", "640:-2")

    active_prefetch: dict[str, Any] | None = None
    prefetch_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch_worker")
        if prefetch_enabled and len(clips) > 1
        else None
    )

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
                    if active_prefetch and active_prefetch.get("clip_id") == get_clip_id(clip):
                        active_prefetch = None
                    if overall_task is not None:
                        progress.advance(overall_task)
                    continue

                clip_cid = get_clip_id(clip)
                prefetched_frames: list[dict] | None = None
                if active_prefetch and active_prefetch.get("clip_id") == clip_cid:
                    try:
                        prefetched_frames = active_prefetch["future"].result()
                    except Exception as exc:
                        logger.warning(
                            "Error waiting for prefetch future for %s: %s", clip_name, exc
                        )
                        prefetched_frames = None
                    active_prefetch = None

                def on_coarse_start_cb(idx=index):
                    nonlocal active_prefetch
                    if prefetch_executor is None:
                        return
                    # Advance to next uncompleted clip when resuming
                    next_idx = idx
                    while next_idx < len(clips):
                        candidate = clips[next_idx]
                        if not (args.resume and candidate.get("file_name") in completed_clips):
                            break
                        next_idx += 1

                    if next_idx >= len(clips):
                        return
                    next_c = clips[next_idx]
                    next_cid = get_clip_id(next_c)
                    next_dir = work_dir / next_cid / "coarse"
                    try:
                        future = prefetch_executor.submit(
                            extract_frames,
                            video_path=next_c["file_path"],
                            output_dir=next_dir,
                            fps=coarse_fps,
                            scale=coarse_scale,
                        )
                        active_prefetch = {"clip_id": next_cid, "future": future}
                    except Exception as exc:
                        logger.warning(
                            "Failed to schedule coarse prefetch for %s: %s",
                            next_c.get("file_name"),
                            exc,
                        )

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
                    checkpoints_dir=checkpoints_dir,
                    resume=args.resume,
                    phase0_stats=p0_stats,
                    prefetched_coarse_frames=prefetched_frames,
                    on_coarse_start=on_coarse_start_cb,
                )
                refined.detections.extend(detections)
                refined.total_detections += len(detections)
                refined.total_clips += 1
                save_detections_atomic(output_path, refined)
                if overall_task is not None:
                    progress.advance(overall_task)
                if state is not None:
                    state.add_progress("scan", clip_name)

                # Clean up intra-clip checkpoint now that clip is committed
                raw_clip_id = clip.get("clip_id") or (
                    f"clip_{hashlib.sha256(clip['file_path'].encode('utf-8')).hexdigest()[:16]}"
                )
                clip_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_clip_id)[:64]
                remove_clip_checkpoint(checkpoints_dir / f"{clip_id}.json")

        # Final atomic write to ensure complete sync
        save_detections_atomic(output_path, refined)

        # Record Ollama calls for each phase
        classify_stats = ollama_client.metrics.by_phase.get("classify")
        if classify_stats:
            p0_stats.ollama_calls = max(p0_stats.ollama_calls, classify_stats.calls)
            p0_stats.ollama_failures = max(p0_stats.ollama_failures, classify_stats.failures)
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
            phase0=p0_stats,
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
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=False, cancel_futures=True)
        cleanup_work_dir(work_dir)


if __name__ == "__main__":
    main()
