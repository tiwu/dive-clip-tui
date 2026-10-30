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
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ffmpeg_streamer import cleanup_work_dir, extract_frames, get_work_dir
from utils.models import DetectionsRefined, RefinedDetection
from utils.ollama_client import OllamaClient
from utils.prompt_templates import ALL_TEMPLATES, get_template_for_keywords


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


def process_clip(
    clip: dict,
    config: dict,
    work_dir: Path,
    ollama_client: OllamaClient | None = None,
) -> list[RefinedDetection]:
    """Process a single clip through coarse-to-fine scanning.

    Returns a list of refined detections (may be empty).
    """
    file_path = clip["file_path"]
    if not Path(file_path).exists():
        print(f"WARNING: File not found: {file_path}")
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
    ollama_host = ollama_cfg.get("host", "http://localhost:11434")
    ollama_model = ollama_cfg.get("model", "qwen2.5-vl:7b")
    ollama_timeout = ollama_cfg.get("timeout_seconds", 90)
    client = ollama_client or OllamaClient(ollama_host, ollama_model, ollama_timeout)

    tags = clip.get("tags", [])

    # ---------- Pass 1: Coarse scan ----------
    print(f"  [Pass 1] Coarse scan @ {coarse_fps} FPS: {clip['file_name']}")
    coarse_frames = extract_frames(
        video_path=file_path,
        output_dir=clip_work_dir / "coarse",
        fps=coarse_fps,
    )

    candidate_windows: list[tuple[float, float]] = []
    for frame_index, frame in enumerate(coarse_frames, start=1):
        print(
            f"  [Pass 1] Analyzing frame {frame_index} of {len(coarse_frames)}",
            flush=True,
        )
        template = get_template_for_keywords(tags)
        prompt_template = template.user_prompt if template else ALL_TEMPLATES[0].user_prompt

        result = call_ollama(
            ollama_host,
            ollama_model,
            frame["frame_path"],
            prompt_template,
            ollama_timeout,
            client=client,
            phase="coarse",
        )
        if result is None:
            continue

        if result.get("detected") and result.get("confidence", 0) >= coarse_threshold:
            ts = frame["timestamp"]
            candidate_windows.append((max(0, ts - fine_padding), ts + fine_padding))

    if not candidate_windows:
        print(f"  [Pass 1] No candidates found for {clip['file_name']}")
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
    print(
        f"  [Pass 2] Fine scan @ {fine_fps} FPS: {clip['file_name']} "
        f"({len(merged)} windows)"
    )
    all_detections: list[RefinedDetection] = []

    template = get_template_for_keywords(tags)
    color = template.color if template else ""
    if template:
        prompt_template = template.user_prompt
    else:
        prompt_template = (
            "Analyze this underwater frame. Describe what you see: subject, "
            "action, sharpness and composition. Return a JSON object with detected, "
            "subject, confidence, quality_score, description and timestamp."
        )
        color = "Green"

    for window_index, (win_start, win_end) in enumerate(merged, start=1):
        duration = win_end - win_start
        win_dir = clip_work_dir / "fine" / f"win_{int(win_start)}_{int(win_end)}"
        fine_frames = extract_frames(
            video_path=file_path,
            output_dir=win_dir,
            fps=fine_fps,
            start_seconds=win_start,
            duration_seconds=duration,
        )

        for frame_index, frame in enumerate(fine_frames, start=1):
            print(
                f"  [Pass 2] Window {window_index} of {len(merged)}: "
                f"analyzing frame {frame_index} of {len(fine_frames)}",
                flush=True,
            )
            result = call_ollama(
                ollama_host,
                ollama_model,
                frame["frame_path"],
                prompt_template,
                ollama_timeout,
                client=client,
                phase="fine",
            )
            if result is None:
                continue

            if result.get("detected") and result.get("quality_score", 0) >= quality_threshold:
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
                            confidence=float(result.get("confidence", 0.0)),
                            quality_score=int(result.get("quality_score", 0)),
                            description=result.get("description", ""),
                            color=color,
                            template=template.name if template else "generic",
                        )
                    )
                except (ValueError, TypeError) as exc:
                    print(f"WARNING: Skipping invalid detection: {exc}")
                    continue

    # Deduplicate overlapping detections (keep highest quality_score)
    all_detections = _deduplicate_detections(all_detections)

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
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("general", {}).get("output_dir", "./outputs"))
    manifest_path = output_dir / "manifest_raw.json"

    if not manifest_path.exists():
        manifest_path = Path(__file__).resolve().parent.parent / "outputs" / "manifest_raw.json"

    if not manifest_path.exists():
        print(f"ERROR: manifest_raw.json not found at {manifest_path}")
        print("Run 00_ingest_media.py --input <folder> to generate it,")
        print("or 01_extract_manifest.py from inside DaVinci Resolve.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    work_dir = get_work_dir()
    print(f"Using work directory: {work_dir}")

    ollama_cfg = config.get("ollama", {})
    ollama_client = OllamaClient(
        ollama_cfg.get("host", "http://localhost:11434"),
        ollama_cfg.get("model", "qwen2.5-vl:7b"),
        ollama_cfg.get("timeout_seconds", 90),
    )
    refined = DetectionsRefined(total_clips=0, total_detections=0, detections=[])

    for clip in manifest.get("clips", []):
        print(f"Processing: {clip.get('file_name', 'unknown')}")
        detections = process_clip(clip, config, work_dir, ollama_client)
        refined.detections.extend(detections)
        refined.total_detections += len(detections)
        refined.total_clips += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detections_refined.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined.model_dump(), f, indent=2, ensure_ascii=False)

    # Clean up the top-level work directory (only removes mkdtemp dirs on macOS;
    # /dev/shm/dive_worker_tmp on Linux is intentionally skipped)
    cleanup_work_dir(work_dir)

    print(
        f"\nDone. {refined.total_detections} detections across "
        f"{refined.total_clips} clips."
    )
    print(ollama_client.metrics.format_summary())
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
