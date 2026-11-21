"""Multi-option storyline prompt engineering, EditPlan validation, and Markdown guide rendering."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from utils.models import EditAct, EditPlan, EditShot, RefinedDetection, StorylineProposalSet
from utils.selection import (
    centered_trim_window,
    make_chronological_sort_key,
    select_diverse_highlights,
    select_duration_budget_highlights,
)

logger = logging.getLogger(__name__)

STORYLINE_SYSTEM_PROMPT = (
    "You are an expert documentary and action-footage video editor. "
    "Given a chronological catalog of AI-detected video segments, design cohesive, "
    "watchable edit proposals of varying lengths. You must ONLY reference real "
    "detection_id values present in the catalog, stay strictly within each detection's "
    "[start_time, end_time] window, avoid repeating the same source file within a "
    "proposal beyond max_per_source_file, and return pure JSON matching the requested schema."
)

STORYLINE_PRESETS: dict[str, dict[str, Any]] = {
    "teaser": {
        "id": "teaser",
        "title": "Fast Teaser Cut",
        "target_duration_seconds": 55,
        "clip_duration_cap": 3.5,
        "min_shot_seconds": 2.0,
        "max_shot_seconds": 3.5,
        "style": "teaser_flashes",
        "instructions": (
            "1-2 brief intro shots (2.0-3.0s) setting the scene, then rapid-fire "
            "2.0-3.5s flashes focused heavily on rare/specialty hero subjects to tease "
            "what appears in a longer cut, closing with a brief 2.5-3.5s outro shot."
        ),
    },
    "highlight": {
        "id": "highlight",
        "title": "5-Act Highlight Aftermovie",
        "target_duration_seconds": 150,
        "clip_duration_cap": 6.5,
        "min_shot_seconds": 3.5,
        "max_shot_seconds": 7.5,
        "style": "story_arc_highlight",
        "instructions": (
            "Balanced 5-act narrative arc (Opening/Transition -> Living Environment -> "
            "Specialty & Macro Highlights -> Climax Encounters -> Resolution/Epilogue) "
            "paced at 4.0-7.0s per shot."
        ),
    },
    "aftermovie": {
        "id": "aftermovie",
        "title": "Extended Trip Aftermovie",
        "target_duration_seconds": 240,
        "clip_duration_cap": 8.0,
        "min_shot_seconds": 4.0,
        "max_shot_seconds": 9.0,
        "style": "story_arc_extended",
        "instructions": (
            "Comprehensive 5-act film covering breadth across days, environments, "
            "human moments, macro details, and signature encounters at 4.5-8.5s per shot."
        ),
    },
}


def _det_field(det: RefinedDetection | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(det, Mapping):
        return det.get(key, default)
    return getattr(det, key, default)


def _det_id(det: RefinedDetection | Mapping[str, Any]) -> str:
    did = str(_det_field(det, "detection_id", "") or "")
    if did:
        return did
    clip_id = str(_det_field(det, "clip_id", ""))
    start_time = float(_det_field(det, "start_time", 0.0))
    end_time = float(_det_field(det, "end_time", start_time))
    return f"{clip_id}_{start_time:08.2f}_{end_time:08.2f}"


def load_clip_metadata(manifest_path: Path | None) -> dict[str, dict[str, Any]]:
    """Load clip metadata keyed by ``file_name`` from ``manifest.json`` if available."""
    if manifest_path is None or not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("Could not read manifest %s: %s", manifest_path, exc)
        return {}
    clips = raw.get("clips", raw) if isinstance(raw, dict) else raw
    if not isinstance(clips, list):
        return {}
    meta: dict[str, dict[str, Any]] = {}
    for clip in clips:
        if isinstance(clip, Mapping) and clip.get("file_name"):
            meta[str(clip["file_name"])] = dict(clip)
    return meta


def load_feedback_data(feedback_path: Path | None) -> dict[str, Any] | None:
    """Load optional ``feedback.json`` from a run directory."""
    if feedback_path is None or not feedback_path.exists():
        return None
    try:
        data = json.loads(feedback_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError) as exc:
        logger.debug("Could not read feedback %s: %s", feedback_path, exc)
        return None


def resolve_preset_specs(
    presets: Sequence[str] | None = None,
    durations: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Resolve requested preset names and/or custom durations into option specifications."""
    if durations:
        specs: list[dict[str, Any]] = []
        preset_names = list(presets) if presets else []
        for idx, dur_s in enumerate(durations):
            target = max(10, int(dur_s))
            if idx < len(preset_names) and preset_names[idx] in STORYLINE_PRESETS:
                base = dict(STORYLINE_PRESETS[preset_names[idx]])
                base["id"] = (
                    f"{base['id']}_{target}s"
                    if target != base["target_duration_seconds"]
                    else base["id"]
                )
            elif target <= 75:
                base = dict(STORYLINE_PRESETS["teaser"])
                base["id"] = f"teaser_{target}s"
            elif target <= 180:
                base = dict(STORYLINE_PRESETS["highlight"])
                base["id"] = f"highlight_{target}s"
            else:
                base = dict(STORYLINE_PRESETS["aftermovie"])
                base["id"] = f"aftermovie_{target}s"
            base["target_duration_seconds"] = target
            base["title"] = f"{base['title']} ({target}s)"
            specs.append(base)
        return specs

    chosen = [
        p.strip().lower() for p in (presets or ["teaser", "highlight", "aftermovie"]) if p.strip()
    ]
    specs = []
    for name in chosen:
        if name in STORYLINE_PRESETS:
            specs.append(dict(STORYLINE_PRESETS[name]))
        else:
            logger.warning("Unknown storyline preset %r; ignoring.", name)
    return specs or [dict(STORYLINE_PRESETS["highlight"])]


def build_storyline_catalog_prompt(
    detections: Sequence[RefinedDetection | Mapping[str, Any]],
    *,
    clip_meta: Mapping[str, Mapping[str, Any]] | None = None,
    preset_specs: Sequence[Mapping[str, Any]] | None = None,
    narrative: str | None = None,
    max_per_source_file: int = 1,
    max_catalog_items: int = 180,
) -> tuple[str, str]:
    """Build ``(system_prompt, user_prompt)`` for multi-option JSON storyline planning."""
    det_list = list(detections)
    if len(det_list) > max_catalog_items:
        det_list = select_diverse_highlights(
            det_list,  # type: ignore[arg-type]
            top_n=max_catalog_items,
            clip_meta=clip_meta,
        )

    ordered = sorted(det_list, key=lambda d: make_chronological_sort_key(d, clip_meta))

    specs = list(preset_specs or resolve_preset_specs())
    lines: list[str] = [
        "Create edit plan proposals for the following options:",
        "",
    ]
    lines.extend(
        f"- Option id={spec['id']!r} | preset={spec.get('style', spec['id'])} | "
        f"target_duration_seconds={spec['target_duration_seconds']} | "
        f"shot_duration={spec.get('min_shot_seconds', 3.5)}-{spec.get('max_shot_seconds', 7.5)}s | "
        f"guidance: {spec['instructions']}"
        for spec in specs
    )

    if narrative and narrative.strip():
        lines.extend(["", f"User creative steering: {narrative.strip()}"])

    lines.extend([
        "",
        "Strict editing rules:",
        "1. Every shot MUST reference an exact 'detection_id' from the catalog below.",
        "2. Set 'in_time' and 'out_time' strictly within that detection's [start_time, end_time] window.",
        f"3. Use each 'file_name' at most {max_per_source_file} time(s) per option (avoid repeating the same video clip in opening and closing).",
        "4. Sequence each option into coherent narrative acts (acts[]), keeping total duration close to target_duration_seconds.",
        "5. Return ONLY a JSON object matching this schema:",
        json.dumps(
            {
                "proposals": [
                    {
                        "id": "teaser",
                        "title": "Short descriptive title",
                        "style": "teaser_flashes",
                        "preset": "teaser",
                        "target_duration_seconds": 55,
                        "narrative_summary": "1-2 sentence overview of the arc",
                        "acts": [
                            {
                                "act_number": 1,
                                "title": "Act 1: Hook",
                                "goal": "Brief visual opener",
                                "shots": [
                                    {
                                        "detection_id": "<exact id from catalog>",
                                        "in_time": 2.0,
                                        "out_time": 5.0,
                                        "rationale": "Why this shot fits here",
                                    }
                                ],
                            }
                        ],
                    }
                ]
            },
            indent=2,
        ),
        "",
        "Chronological catalog of candidate detections (grouped by date):",
    ])

    by_day: dict[str, list[Any]] = defaultdict(list)
    for det in ordered:
        fname = str(_det_field(det, "file_name", ""))
        ctime = str((clip_meta or {}).get(fname, {}).get("creation_time") or "")
        day = ctime[:10] if len(ctime) >= 10 else "undated"
        by_day[day].append((ctime, det))

    for day in sorted(by_day):
        lines.append(f"\n## Date: {day}")
        for ctime, det in by_day[day]:
            did = _det_id(det)
            fname = str(_det_field(det, "file_name", ""))
            st = float(_det_field(det, "start_time", 0.0))
            et = float(_det_field(det, "end_time", st))
            dur = max(0.0, et - st)
            domain = str(_det_field(det, "domain", "") or "generic")
            tpl = str(_det_field(det, "template", "") or "")
            role = str(_det_field(det, "story_role", "hero_subject") or "hero_subject")
            q = int(_det_field(det, "quality_score", 0) or 0)
            subj = str(_det_field(det, "subject", "") or "")
            desc = str(_det_field(det, "description", "") or "").replace("\n", " ")[:160]
            time_tag = ctime[11:16] if len(ctime) >= 16 else "n/a"
            lines.append(
                f"- detection_id={did} | file={fname} | time={time_tag} | "
                f"window={st:.1f}-{et:.1f}s ({dur:.1f}s) | domain={domain}/{tpl} | "
                f"role={role} | q={q}/10 | subject={subj} | desc={desc}"
            )

    return STORYLINE_SYSTEM_PROMPT, "\n".join(lines)


def validate_and_normalize_plan(
    raw_plan: Mapping[str, Any] | EditPlan,
    detections_by_id: Mapping[str, RefinedDetection | Mapping[str, Any]],
    clip_meta: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    max_per_source_file: int = 1,
    min_shot_duration: float = 1.5,
    default_clip_cap: float = 6.5,
) -> tuple[EditPlan, list[str]]:
    """Validate an LLM or manual edit plan against real detections and constraints.

    Drops hallucinated ``detection_id`` entries, clamps ``[in_time, out_time]``
    strictly inside the source detection bounds, enforces ``max_per_source_file``,
    and stamps act-boundary transitions (``fade`` at act boundaries, ``cut`` within acts).
    """
    plan_dict = raw_plan.model_dump() if isinstance(raw_plan, EditPlan) else dict(raw_plan)
    warnings: list[str] = []

    raw_acts = plan_dict.get("acts")
    if not isinstance(raw_acts, list) or not raw_acts:
        flat_shots = plan_dict.get("shots")
        raw_acts = (
            [{"act_number": 1, "title": "Act 1: Sequence", "goal": "", "shots": flat_shots}]
            if isinstance(flat_shots, list)
            else []
        )

    plan_id = str(plan_dict.get("id") or "custom_plan")
    source_counts: Counter[str] = Counter()
    seen_det_ids: set[str] = set()

    normalized_act_shots: list[tuple[str, str, list[EditShot]]] = []
    for act_idx, raw_act in enumerate(raw_acts, start=1):
        if not isinstance(raw_act, Mapping):
            continue
        act_title = str(raw_act.get("title") or f"Act {act_idx}")
        act_goal = str(raw_act.get("goal") or "")
        shot_entries = raw_act.get("shots")
        if not isinstance(shot_entries, list):
            continue

        kept_shots: list[EditShot] = []
        for raw_shot in shot_entries:
            if not isinstance(raw_shot, Mapping):
                continue
            did = str(raw_shot.get("detection_id") or "").strip()
            if not did or did not in detections_by_id:
                warnings.append(f"[{plan_id}] Dropped unknown detection_id: {did!r}")
                continue
            if did in seen_det_ids:
                warnings.append(f"[{plan_id}] Dropped duplicate detection_id: {did}")
                continue

            det = detections_by_id[did]
            fname = str(_det_field(det, "file_name", ""))
            clip_id = str(_det_field(det, "clip_id", ""))
            src_key = fname or clip_id
            if max_per_source_file >= 1 and source_counts[src_key] >= max_per_source_file:
                warnings.append(
                    f"[{plan_id}] Dropped {did} ({fname}): exceeded max_per_source_file={max_per_source_file}"
                )
                continue

            det_start = float(_det_field(det, "start_time", 0.0))
            det_end = float(_det_field(det, "end_time", det_start))
            try:
                raw_in = float(raw_shot.get("in_time", det_start))
                raw_out = float(raw_shot.get("out_time", det_end))
            except TypeError, ValueError:
                raw_in, raw_out = det_start, det_end

            in_t = round(max(det_start, min(raw_in, det_end)), 2)
            out_t = round(min(det_end, max(raw_out, det_start)), 2)
            if out_t - in_t < min_shot_duration:
                in_t, out_t = centered_trim_window(det_start, det_end, default_clip_cap)
                if out_t - in_t < min_shot_duration or out_t <= in_t:
                    warnings.append(
                        f"[{plan_id}] Dropped {did}: window {det_start:.1f}-{det_end:.1f}s shorter than {min_shot_duration}s"
                    )
                    continue

            ctime = str((clip_meta or {}).get(fname, {}).get("creation_time") or "")
            shot = EditShot(
                detection_id=did,
                clip_id=clip_id,
                file_name=fname,
                file_path=str(_det_field(det, "file_path", "")),
                creation_time=ctime,
                act=1,  # renumbered below after filtering empty acts
                act_title=act_title,
                in_time=in_t,
                out_time=out_t,
                duration=round(out_t - in_t, 2),
                subject=str(_det_field(det, "subject", "") or ""),
                story_role=str(_det_field(det, "story_role", "hero_subject") or "hero_subject"),
                quality_score=int(_det_field(det, "quality_score", 0) or 0),
                transition_in="cut",
                transition_out="cut",
                rationale=str(raw_shot.get("rationale") or ""),
            )
            kept_shots.append(shot)
            seen_det_ids.add(did)
            source_counts[src_key] += 1

        if kept_shots:
            normalized_act_shots.append((act_title, act_goal, kept_shots))

    # Renumber acts 1..K and stamp act-boundary transitions
    acts: list[EditAct] = []
    flat_shots: list[EditShot] = []
    for act_num, (act_title, act_goal, act_shots) in enumerate(normalized_act_shots, start=1):
        updated_act_shots: list[EditShot] = []
        for idx, shot in enumerate(act_shots):
            updated = shot.model_copy(
                update={
                    "act": act_num,
                    "act_title": act_title,
                    "transition_in": "fade" if idx == 0 else "cut",
                    "transition_out": "fade" if idx == len(act_shots) - 1 else "cut",
                }
            )
            updated_act_shots.append(updated)
            flat_shots.append(updated)
        acts.append(
            EditAct(
                act_number=act_num,
                title=act_title,
                goal=act_goal,
                shots=updated_act_shots,
            )
        )

    total_dur = round(sum(s.duration for s in flat_shots), 2)
    target_dur = int(plan_dict.get("target_duration_seconds") or max(1, round(total_dur)) or 60)
    plan = EditPlan(
        id=plan_id,
        title=str(plan_dict.get("title") or plan_id),
        style=str(plan_dict.get("style") or "story_arc"),
        preset=str(plan_dict.get("preset") or plan_id),
        target_duration_seconds=max(1, target_dur),
        estimated_duration_seconds=total_dur,
        clip_count=len(flat_shots),
        narrative_summary=str(plan_dict.get("narrative_summary") or ""),
        acts=acts,
        shots=flat_shots,
    )
    return plan, warnings


def generate_deterministic_proposals(
    detections: Sequence[RefinedDetection | dict[str, Any]],
    *,
    clip_meta: Mapping[str, Mapping[str, Any]] | None = None,
    feedback: Mapping[str, Any] | None = None,
    preset_specs: Sequence[Mapping[str, Any]] | None = None,
    max_per_source_file: int = 1,
    run_id: str = "",
    source_detections: str = "",
    outlier_days: int | None = 7,
    allow_outlier_dates: bool = False,
    date_range: tuple[str, str] | None = None,
    exclude_files: Sequence[str] | None = None,
) -> tuple[StorylineProposalSet, dict[str, list[str]]]:
    """Generate deterministic multi-option ``StorylineProposalSet`` without calling an LLM."""
    specs = list(preset_specs or resolve_preset_specs())
    proposals: list[EditPlan] = []
    combined_report: dict[str, list[str]] = {}

    for spec in specs:
        preset_key = "teaser" if "teaser" in str(spec.get("style", "")) else str(spec["id"])
        plan, report = select_duration_budget_highlights(
            detections,
            clip_meta=clip_meta,
            feedback=feedback,
            target_duration=float(spec["target_duration_seconds"]),
            clip_duration_cap=float(spec.get("clip_duration_cap", 6.5)),
            max_per_source_file=max_per_source_file,
            order="story_arc",
            preset=preset_key,
            plan_id=str(spec["id"]),
            title=str(spec["title"]),
            outlier_days=outlier_days,
            allow_outlier_dates=allow_outlier_dates,
            date_range=date_range,
            exclude_files=exclude_files,
        )
        proposals.append(plan)
        if not combined_report:
            combined_report = report

    proposal_set = StorylineProposalSet(
        schema_version=1,
        run_id=run_id,
        provider="deterministic",
        model="mmr-story-arc",
        source_detections=source_detections,
        max_per_source_file=max(1, max_per_source_file),
        proposals=proposals,
    )
    return proposal_set, combined_report


def render_storyline_markdown_guide(
    proposal_set: StorylineProposalSet,
    plans_filename: str = "storyline_plans.json",
) -> str:
    """Render a human-readable Markdown edit guide from a validated ``StorylineProposalSet``."""
    lines: list[str] = [
        "# Storyline Edit Guide",
        "",
        f"- **Run:** `{proposal_set.run_id or 'current'}`",
        f"- **Planner:** `{proposal_set.provider}` (`{proposal_set.model or 'n/a'}`)",
        f"- **Max per source file:** `{proposal_set.max_per_source_file}`",
        "",
        "## Options Overview",
        "",
        "| Option ID | Title | Target | Estimated | Clips | Acts | Render Command |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for plan in proposal_set.proposals:
        cmd = f"`dive-clip aftermovie --plan {plans_filename} --option {plan.id}`"
        lines.append(
            f"| `{plan.id}` | {plan.title} | {plan.target_duration_seconds}s | "
            f"{plan.estimated_duration_seconds:.1f}s | {plan.clip_count} | "
            f"{len(plan.acts)} | {cmd} |"
        )

    for plan in proposal_set.proposals:
        lines.extend([
            "",
            "---",
            "",
            f"## Option `{plan.id}` — {plan.title}",
            "",
            f"- **Estimated runtime:** {plan.estimated_duration_seconds:.1f}s "
            f"(target {plan.target_duration_seconds}s across {plan.clip_count} shots)",
        ])
        if plan.narrative_summary:
            lines.append(f"- **Summary:** {plan.narrative_summary}")

        for act in plan.acts:
            lines.extend(["", f"### {act.title}"])
            if act.goal:
                lines.append(f"*{act.goal}*")
            lines.extend([
                "",
                "| # | File | Window | Dur | Role | Subject | Transitions | Rationale |",
                "|---:|---|---|---:|---|---|---|---|",
            ])
            for idx, shot in enumerate(act.shots, start=1):
                lines.append(
                    f"| {idx} | `{shot.file_name}` | `{shot.in_time:.1f}s-{shot.out_time:.1f}s` | "
                    f"{shot.duration:.1f}s | `{shot.story_role}` | {shot.subject} | "
                    f"{shot.transition_in}/{shot.transition_out} | {shot.rationale} |"
                )

    lines.append("")
    return "\n".join(lines)
