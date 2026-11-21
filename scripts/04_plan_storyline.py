"""Multi-option storyline planner (Ollama, OpenRouter, Claude, Gemini, or deterministic).

Reads detections_refined.json (plus manifest.json and optional feedback.json),
generates multiple validated edit proposals (teaser, highlight, aftermovie, or custom
durations) into storyline_plans.json + storyline_edit_guide.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import ConfigError, load_config
from utils.logging_setup import configure_logging, get_console
from utils.models import DetectionsRefined, EditPlan, StorylineProposalSet
from utils.run_archive import archive_run_artifacts, resolve_assembly_artifacts
from utils.selection import filter_selection_candidates, select_diverse_highlights
from utils.signals import setup_signal_handlers
from utils.storyline import (
    build_storyline_catalog_prompt,
    generate_deterministic_proposals,
    load_clip_metadata,
    load_feedback_data,
    render_storyline_markdown_guide,
    resolve_preset_specs,
    validate_and_normalize_plan,
)
from utils.text_llm import (
    ANTHROPIC_API_KEY_ENV_VAR,
    GEMINI_API_KEY_ENV_VAR,
    TextLLMError,
    create_text_llm_client,
    resolve_storyline_llm_settings,
)
from utils.vlm_common import extract_json_object

logger = logging.getLogger(__name__)


def build_prompt(detections: DetectionsRefined, target_duration: int = 4) -> str:
    """Build the legacy single-option storyline prompt (preserved for compatibility).

    Truncates detections to the highest-quality ones if the prompt would
    exceed a safe token budget (~8,000 tokens ≈ 32,000 chars).
    """
    max_detections = 200
    det_list = detections.detections
    if len(det_list) > max_detections:
        det_list = select_diverse_highlights(det_list, top_n=max_detections)

    lines = [
        "You are a professional underwater video editor and storyteller.",
        "You get a list of AI-detected highlights from underwater dive footage.",
        "",
        f"Goal: create an edit plan of approximately {target_duration} minutes.",
        "",
        "Detections (per clip):",
    ]

    for i, det in enumerate(det_list, 1):
        lines.append(
            f"{i}. {det.file_name} | {det.start_time:.1f}s-{det.end_time:.1f}s | "
            f"{det.subject} | quality {det.quality_score}/10 | confidence {det.confidence:.2f} | "
            f"{det.description}"
        )

    lines.extend([
        "",
        "Requirements for the edit plan:",
        "1. Determine a logical narrative arc (Establishing -> Reef Scape -> Encounters -> Macro Treasures -> Safety Stop).",
        "2. Provide a recommended shot order with timecodes.",
        "3. Suggest music and color grading per segment.",
        "4. Mark which shots can serve as B-roll.",
        "5. Take into account the colored markers (Cyan=pelagic, Yellow=macro, Green=diver/composition).",
        "",
        "Return the result as markdown with a clear timeline order.",
    ])
    return "\n".join(lines)


def call_claude(api_key: str, model: str, prompt: str) -> str:
    """Call Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt, request_options={"timeout": 120})
    return response.text or ""


def _parse_date_range(raw: str | None) -> tuple[str, str] | None:
    if not raw or not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(":", 1)]
    if len(parts) != 2:
        raise ValueError(f"Invalid --date-range {raw!r}; expected YYYY-MM-DD:YYYY-MM-DD")
    return parts[0], parts[1]


def _print_proposals_table(proposal_set: StorylineProposalSet, plans_path: Path) -> None:
    console = get_console()
    if not console.is_terminal:
        return
    table = Table(
        title=(f"Storyline Proposals ({proposal_set.provider}: {proposal_set.model or 'auto'})")
    )
    table.add_column("Option ID", style="cyan bold")
    table.add_column("Title", style="white")
    table.add_column("Target", justify="right")
    table.add_column("Estimated", justify="right", style="green")
    table.add_column("Clips", justify="right")
    table.add_column("Acts", justify="right")
    for plan in proposal_set.proposals:
        table.add_row(
            plan.id,
            plan.title,
            f"{plan.target_duration_seconds}s",
            f"{plan.estimated_duration_seconds:.1f}s",
            str(plan.clip_count),
            str(len(plan.acts)),
        )
    console.print(table)
    if proposal_set.proposals:
        first_id = proposal_set.proposals[0].id
        console.print(
            f'[dim]Preview a plan:[/] dive-clip aftermovie --plan "{plans_path}" '
            f"--option {first_id} --dry-run"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-option storyline planner (Ollama / OpenRouter / Claude / Gemini / deterministic)"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--run",
        default=None,
        metavar="RUN_ID_OR_PATH",
        help="Run ID or path (defaults to latest run)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to detections_refined.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination path for storyline_edit_guide.md",
    )
    parser.add_argument(
        "--plans-output",
        default=None,
        help="Destination path for storyline_plans.json",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Text LLM provider override: ollama, openrouter, claude, or gemini",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override for the selected provider",
    )
    parser.add_argument(
        "--presets",
        default=None,
        help="Comma-separated preset names (default: teaser,highlight,aftermovie)",
    )
    parser.add_argument(
        "--durations",
        default=None,
        help="Comma-separated target durations in seconds (e.g. 60,150,240)",
    )
    parser.add_argument(
        "--narrative",
        "--prompt",
        dest="narrative",
        default=None,
        help="Optional narrative steering instructions",
    )
    parser.add_argument(
        "--max-per-source-file",
        type=int,
        default=None,
        help="Maximum clips allowed per source video file within a proposal (default: 1)",
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
        help="Disable automatic exclusion of off-trip outlier dates",
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        default=False,
        help="Generate storyline_plans.json locally without calling any LLM",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--trace", action="store_true", help="Enable trace logging")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional logfile path")
    args = parser.parse_args()
    setup_signal_handlers()
    configure_logging(debug=args.debug, trace=args.trace, log_file=args.log_file)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    output_dir = Path(config.get("general", {}).get("output_dir", "./outputs"))
    try:
        detections_path, _, latest_run = resolve_assembly_artifacts(
            output_dir,
            input_detections=args.input,
            run=args.run,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if not detections_path.exists():
        alt = Path(__file__).resolve().parent.parent / "outputs" / "detections_refined.json"
        if alt.exists():
            detections_path = alt
    if not detections_path.exists():
        logger.error(
            "detections_refined.json not found at %s. Run 02_scan_footage.py first.",
            detections_path,
        )
        sys.exit(1)

    try:
        with open(detections_path, encoding="utf-8") as f:
            data = json.load(f)
        detections = DetectionsRefined(**data)
    except (OSError, ValueError, ValidationError) as exc:
        logger.error("Invalid detections_refined.json: %s", exc)
        sys.exit(1)

    storyline_cfg = config.get("storyline") if isinstance(config.get("storyline"), dict) else {}
    max_per_source_file = (
        args.max_per_source_file
        if args.max_per_source_file is not None
        else int(storyline_cfg.get("max_per_source_file", 1))
    )
    outlier_days = int(storyline_cfg.get("outlier_days", 7))
    try:
        date_range = _parse_date_range(args.date_range)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    preset_list = (
        [p.strip() for p in args.presets.split(",") if p.strip()]
        if args.presets
        else storyline_cfg.get("presets", ["teaser", "highlight", "aftermovie"])
    )
    duration_list: list[int] | None = None
    if args.durations:
        try:
            duration_list = [int(d.strip()) for d in args.durations.split(",") if d.strip()]
        except ValueError:
            logger.error(
                "Invalid --durations value: %r (expected comma-separated integers)", args.durations
            )
            sys.exit(1)
    preset_specs = resolve_preset_specs(presets=preset_list, durations=duration_list)

    run_base = (
        latest_run
        if latest_run is not None and detections_path.parent.resolve() == latest_run.resolve()
        else detections_path.parent
    )
    manifest_path = run_base / "manifest.json"
    feedback_path = run_base / "feedback.json"
    clip_meta = load_clip_metadata(manifest_path)
    feedback = load_feedback_data(feedback_path)

    guide_path = Path(args.output) if args.output else (run_base / "storyline_edit_guide.md")
    plans_path = (
        Path(args.plans_output)
        if args.plans_output
        else (guide_path.parent / "storyline_plans.json")
    )

    # Validate provider and credentials before skipping empty detections so CLI checks are deterministic.
    provider = "deterministic"
    model = "mmr-story-arc"
    llm_settings: dict[str, Any] = {"temperature": 0.4, "timeout_seconds": 120, "max_tokens": 4096}
    if not args.deterministic_only:
        try:
            provider, model, llm_settings = resolve_storyline_llm_settings(
                config,
                provider_override=args.provider,
                model_override=args.model,
            )
            if provider == "claude" and not os.environ.get(ANTHROPIC_API_KEY_ENV_VAR, "").strip():
                raise TextLLMError(f"{ANTHROPIC_API_KEY_ENV_VAR} environment variable not set.")
            if provider == "gemini" and not os.environ.get(GEMINI_API_KEY_ENV_VAR, "").strip():
                raise TextLLMError(f"{GEMINI_API_KEY_ENV_VAR} environment variable not set.")
            if provider == "openrouter":
                create_text_llm_client(config, provider=provider, model=model)
        except TextLLMError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    filtered_dets, _filter_report = filter_selection_candidates(
        detections.detections,
        clip_meta=clip_meta,
        feedback=feedback,
        outlier_days=outlier_days,
        allow_outlier_dates=args.allow_outlier_dates,
        date_range=date_range,
        exclude_files=args.exclude_file,
    )
    det_by_id = {d.detection_id: d for d in filtered_dets}

    fallback_set, _ = generate_deterministic_proposals(
        detections.detections,
        clip_meta=clip_meta,
        feedback=feedback,
        preset_specs=preset_specs,
        max_per_source_file=max_per_source_file,
        run_id=latest_run.name if latest_run is not None else "",
        source_detections=str(detections_path),
        outlier_days=outlier_days,
        allow_outlier_dates=args.allow_outlier_dates,
        date_range=date_range,
        exclude_files=args.exclude_file,
    )
    fallback_by_id = {p.id: p for p in fallback_set.proposals}

    proposal_set = fallback_set
    raw_markdown_override: str | None = None

    if not args.deterministic_only and filtered_dets:
        sys_prompt, user_prompt = build_storyline_catalog_prompt(
            filtered_dets,
            clip_meta=clip_meta,
            preset_specs=preset_specs,
            narrative=args.narrative,
            max_per_source_file=max_per_source_file,
        )
        parsed_json: dict[str, Any] | None = None
        if provider == "claude":
            api_key = os.environ.get(ANTHROPIC_API_KEY_ENV_VAR, "")
            raw_text = call_claude(api_key, model, f"{sys_prompt}\n\n{user_prompt}")
            parsed_json = extract_json_object(raw_text)
            if parsed_json is None and raw_text.strip():
                raw_markdown_override = raw_text
        elif provider == "gemini":
            api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR, "")
            raw_text = call_gemini(api_key, model, f"{sys_prompt}\n\n{user_prompt}")
            parsed_json = extract_json_object(raw_text)
            if parsed_json is None and raw_text.strip():
                raw_markdown_override = raw_text
        else:
            try:
                client = create_text_llm_client(config, provider=provider, model=model)
                parsed_json = client.generate_json(
                    user_prompt,
                    system_prompt=sys_prompt,
                    temperature=llm_settings["temperature"],
                    max_tokens=llm_settings["max_tokens"],
                )
            except TextLLMError as exc:
                logger.error("%s", exc)
                sys.exit(1)

        if isinstance(parsed_json, dict) and isinstance(parsed_json.get("proposals"), list):
            validated_plans: list[EditPlan] = []
            for raw_plan in parsed_json["proposals"]:
                if not isinstance(raw_plan, dict):
                    continue
                norm_plan, plan_warnings = validate_and_normalize_plan(
                    raw_plan,
                    det_by_id,
                    clip_meta=clip_meta,
                    max_per_source_file=max_per_source_file,
                )
                for w in plan_warnings:
                    logger.warning("%s", w)
                if norm_plan.clip_count > 0:
                    validated_plans.append(norm_plan)
                elif norm_plan.id in fallback_by_id:
                    logger.warning(
                        "LLM proposal %r had 0 valid shots; using deterministic fallback.",
                        norm_plan.id,
                    )
                    validated_plans.append(fallback_by_id[norm_plan.id])

            if validated_plans:
                proposal_set = StorylineProposalSet(
                    schema_version=1,
                    run_id=latest_run.name if latest_run is not None else "",
                    provider=provider,
                    model=model,
                    source_detections=str(detections_path),
                    max_per_source_file=max(1, max_per_source_file),
                    proposals=validated_plans,
                )

    plans_path.parent.mkdir(parents=True, exist_ok=True)
    plans_path.write_text(proposal_set.model_dump_json(indent=2), encoding="utf-8")

    guide_text = (
        raw_markdown_override
        if raw_markdown_override is not None
        else render_storyline_markdown_guide(proposal_set, plans_filename=plans_path.name)
    )
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    guide_path.write_text(guide_text, encoding="utf-8")

    if latest_run is not None:
        archive_run_artifacts(
            latest_run,
            storyline_plans_path=plans_path,
            storyline_guide_path=guide_path,
        )

    _print_proposals_table(proposal_set, plans_path)
    logger.info("Storyline plans written: %s", plans_path)
    logger.info("Storyline edit guide written: %s", guide_path)


if __name__ == "__main__":
    main()
