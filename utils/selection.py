"""Maximal Marginal Relevance (MMR) highlight selection and shot tokenization."""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from utils.models import (
    EditAct,
    EditPlan,
    EditShot,
    RefinedDetection,
    SelectionCandidateTrace,
    SelectionTraceOutput,
)
from utils.prescreen import hsv_similarity

logger = logging.getLogger(__name__)

# Blend factor for the optional HSV histogram signal in pairwise similarity.
# Text-only behaviour is preserved exactly when either histogram is missing.
HSV_SIMILARITY_WEIGHT = 0.25

STOPWORDS: frozenset[str] = frozenset({
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "over",
    "under",
    "near",
    "through",
    "while",
    "some",
    "two",
    "three",
    "clear",
    "good",
    "shot",
    "view",
    "frame",
    "scene",
    "clip",
    "video",
    "camera",
    "action",
    "highlight",
    "swimming",
    "moving",
    "riding",
    "looking",
    "showing",
    "visible",
    "water",
    "underwater",
    "road",
    "unknown",
    "none",
})


_GOPRO_NAME_RE = re.compile(r"^G[XHPA](\d{2})(\d{4})", re.IGNORECASE)

BOGUS_SUBJECTS: frozenset[str] = frozenset({
    "",
    "none",
    "no diver visible",
    "no subject",
})

GENERIC_RARITY_STOPWORDS: frozenset[str] = frozenset({
    "coral",
    "corals",
    "reef",
    "fish",
    "sea",
    "diver",
    "exploring",
    "school",
    "yellow",
    "blue",
    "orange",
    "small",
    "large",
    "unknown",
    "broad",
    "marine",
    "spotted",
    "striped",
    "spp",
})

MACRO_SUBJECT_KEYWORDS: tuple[str, ...] = (
    "nudibranch",
    "hypselodoris",
    "goniobranchus",
    "seahorse",
    "frogfish",
    "clownfish",
    "anemonefish",
    "anemone",
    "pipefish",
    "stonefish",
    "lionfish",
    "octopus",
    "lobster",
    "shrimp",
    "crab",
    "starfish",
    "clam",
    "tridacna",
    "sponge",
    "sand dollar",
    "flatfish",
    "cucumber",
)

ASCENT_SUBJECT_KEYWORDS: tuple[str, ...] = (
    "ascend",
    "ascending",
    "bubble stream",
    "bubble trail",
    "surface",
    "sunlit background",
)

ACT_TITLES: dict[int, str] = {
    1: "Act 1: Arrival & Setting Sail",
    2: "Act 2: Descending onto the Living Reef",
    3: "Act 3: Macro Treasures & Hidden Critters",
    4: "Act 4: Big Encounters & Drama",
    5: "Act 5: Ascent, Surface & Epilogue",
}


def tokenize_content_words(text: str | None) -> frozenset[str]:
    """Extract content tokens of length >= 3 excluding deterministic stopwords."""
    if not text:
        return frozenset()
    raw_tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return frozenset(t for t in raw_tokens if len(t) >= 3 and t not in STOPWORDS)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve attribute from dict or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_val(obj: Any, key: str, value: Any) -> Any:
    """Assign attribute or dict key, returning updated object."""
    if isinstance(obj, dict):
        obj[key] = value
        return obj
    try:
        setattr(obj, key, value)
    except Exception:
        # If frozen or model attribute assignment fails, try model_copy
        if hasattr(obj, "model_copy"):
            return obj.model_copy(update={key: value})
    return obj


def calculate_intrinsic_relevance(quality_score: int | float, confidence: float) -> float:
    """Calculate R(d) in [0.0, 1.0] from quality score and confidence."""
    q_clamped = max(1.0, min(10.0, float(quality_score)))
    c_clamped = max(0.0, min(1.0, float(confidence)))
    return round(0.85 * (q_clamped / 10.0) + 0.15 * c_clamped, 6)


def _jaccard(tokens_a: frozenset[str], tokens_b: frozenset[str]) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return intersection / union


def _optional_float(value: Any) -> float | None:
    """Coerce a value to float, returning None when it is missing or unusable."""
    if value is None:
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _histogram_similarity(d_i: Any, d_j: Any) -> float | None:
    """Return HSV histogram similarity for two detections, or None when unavailable."""
    left = _get_val(d_i, "hsv_histogram", None) or []
    right = _get_val(d_j, "hsv_histogram", None) or []
    if not left or not right:
        return None
    return hsv_similarity(left, right)


def calculate_pairwise_similarity(d_i: Any, d_j: Any) -> float:
    """Calculate S(d_i, d_j) in [0.0, 1.0] across subject, description, category, and time.

    When both detections carry an HSV histogram, the text score is blended with
    histogram-intersection similarity. Missing histograms leave the text-only
    score untouched so existing behaviour is preserved.
    """
    subj_i = str(_get_val(d_i, "subject", "") or "").strip().lower()
    subj_j = str(_get_val(d_j, "subject", "") or "").strip().lower()

    desc_i = str(_get_val(d_i, "description", "") or "")
    desc_j = str(_get_val(d_j, "description", "") or "")
    desc_tokens_i = tokenize_content_words(desc_i)
    desc_tokens_j = tokenize_content_words(desc_j)
    j_desc = _jaccard(desc_tokens_i, desc_tokens_j)

    if subj_i and subj_i not in {"none", "unknown"} and subj_i == subj_j:
        j_subj = 1.0
    else:
        subj_tokens_i = tokenize_content_words(subj_i)
        subj_tokens_j = tokenize_content_words(subj_j)
        # When a label is "unknown" or yields no content tokens, fall back to
        # description tokens so unlabeled scenes cannot bypass diversity penalties.
        eff_i = subj_tokens_i or desc_tokens_i
        eff_j = subj_tokens_j or desc_tokens_j
        j_subj = _jaccard(eff_i, eff_j)

    cat_i = (
        str(_get_val(d_i, "template", "") or _get_val(d_i, "story_role", "") or "").strip().lower()
    )
    cat_j = (
        str(_get_val(d_j, "template", "") or _get_val(d_j, "story_role", "") or "").strip().lower()
    )
    i_cat = 1.0 if (cat_i and cat_j and cat_i == cat_j and cat_i != "none") else 0.0

    clip_i = str(_get_val(d_i, "clip_id", ""))
    clip_j = str(_get_val(d_j, "clip_id", ""))
    if clip_i and clip_i == clip_j:
        start_i = float(_get_val(d_i, "start_time", 0.0))
        end_i = float(_get_val(d_i, "end_time", start_i))
        mid_i = (start_i + end_i) / 2.0

        start_j = float(_get_val(d_j, "start_time", 0.0))
        end_j = float(_get_val(d_j, "end_time", start_j))
        mid_j = (start_j + end_j) / 2.0

        diff = abs(mid_i - mid_j)
        t_prox = max(0.0, 1.0 - diff / 60.0)
    else:
        t_prox = 0.0

    similarity = 0.50 * j_subj + 0.20 * j_desc + 0.15 * i_cat + 0.15 * t_prox
    hist_sim = _histogram_similarity(d_i, d_j)
    if hist_sim is not None:
        similarity = (1.0 - HSV_SIMILARITY_WEIGHT) * similarity + HSV_SIMILARITY_WEIGHT * hist_sim
    return max(0.0, min(1.0, similarity))


def make_detection_id(clip_id: str, start_time: float, end_time: float) -> str:
    """Build deterministic detection identifier."""
    return f"{clip_id}_{start_time:08.2f}_{end_time:08.2f}"


def make_sort_key(source_file: str, start_time: float, end_time: float, clip_id: str) -> str:
    """Build canonical chronological sort key."""
    return f"{source_file}:{start_time:08.2f}:{end_time:08.2f}:{clip_id}"


def parse_gopro_filename(file_name: str) -> tuple[int, int, int, str]:
    """Return a sort tuple placing GoPro files in ``(file_num, chapter)`` order.

    Ensures multi-chapter recordings like ``GX020874.MP4`` sort before
    ``GX010876.MP4`` even when ``creation_time`` is identical or unavailable.
    """
    match = _GOPRO_NAME_RE.match(file_name)
    if match:
        chapter = int(match.group(1))
        file_num = int(match.group(2))
        return (0, file_num, chapter, file_name)
    return (1, 0, 0, file_name)


def make_chronological_sort_key(
    det: Any,
    clip_meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[str, tuple[int, int, int, str], float, float, str]:
    """Return a true chronological sort key using ``creation_time`` and GoPro chapters."""
    source_file = str(_get_val(det, "file_name", "") or _get_val(det, "source_file", ""))
    ctime = ""
    if clip_meta and source_file in clip_meta:
        ctime = str(clip_meta[source_file].get("creation_time") or "")
    if not ctime:
        ctime = str(_get_val(det, "creation_time", "") or "")
    start_time = float(_get_val(det, "start_time", 0.0))
    end_time = float(_get_val(det, "end_time", start_time))
    clip_id = str(_get_val(det, "clip_id", ""))
    return (
        ctime if ctime else "9999",
        parse_gopro_filename(source_file),
        start_time,
        end_time,
        clip_id,
    )


def select_diverse_highlights[T: (RefinedDetection, dict[str, Any])](
    detections: list[T],
    top_n: int,
    lambda_quality: float = 0.65,
    max_per_clip: int | None = None,
    trace_path: Path | str | None = None,
    *,
    clip_meta: Mapping[str, Mapping[str, Any]] | None = None,
    max_per_source_file: int | None = None,
) -> list[T]:
    """Select up to top_n diverse highlights using Maximal Marginal Relevance (MMR).

    Returns selected detections sorted chronologically by true recording timestamp
    (when ``clip_meta`` is supplied) and GoPro chapter order, falling back to
    ``(source_file, start_time, end_time, clip_id)``.
    """
    if not detections or top_n <= 0:
        if trace_path:
            out = SelectionTraceOutput(
                top_n=top_n,
                lambda_quality=lambda_quality,
                per_clip_limit=1,
                total_candidates=len(detections) if detections else 0,
                selected_count=0,
                candidates=[],
            )
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            Path(trace_path).write_text(out.model_dump_json(indent=2), encoding="utf-8")
        return []

    # Calculate per-clip quota limit
    unique_clips = len({str(_get_val(x, "clip_id", "")) for x in detections})
    per_clip_limit = (
        max_per_clip
        if max_per_clip is not None
        else max(1, math.ceil(top_n / max(1, unique_clips)))
    )

    # Prepare candidates metadata
    candidate_indices = list(range(len(detections)))
    selected_indices: list[int] = []

    # Track decision metadata per candidate
    trace_details: dict[int, dict[str, Any]] = {}

    for idx in candidate_indices:
        d = detections[idx]
        q = _get_val(d, "quality_score", 0)
        c = _get_val(d, "confidence", 0.0)
        r = calculate_intrinsic_relevance(q, c)
        clip_id = str(_get_val(d, "clip_id", ""))
        source_file = str(_get_val(d, "file_name", "") or _get_val(d, "source_file", ""))
        start_time = float(_get_val(d, "start_time", 0.0))
        end_time = float(_get_val(d, "end_time", start_time))

        trace_details[idx] = {
            "detection_id": make_detection_id(clip_id, start_time, end_time),
            "clip_id": clip_id,
            "source_file": source_file,
            "start_time": start_time,
            "end_time": end_time,
            "sort_key": make_sort_key(source_file, start_time, end_time, clip_id),
            "subject": str(_get_val(d, "subject", "")),
            "story_role": str(_get_val(d, "story_role", "hero_subject")),
            "template": str(_get_val(d, "template", "")),
            "quality_score": int(q),
            "confidence": float(c),
            "relevance_score": r,
            "max_similarity_penalty": 0.0,
            "most_similar_selected_id": None,
            "quota_penalty": 0.0,
            "mmr_score": 0.0,
            "selected": False,
            "selection_rank": None,
            "chronological_rank": None,
            "decision_reason": "unselected candidate",
            "telemetry_score": _optional_float(_get_val(d, "telemetry_score", None)),
            "hsv_similarity": None,
        }

    k_target = min(top_n, len(detections))

    for step in range(1, k_target + 1):
        best_idx: int | None = None
        best_score = -float("inf")
        best_sim_penalty = 0.0
        best_similar_id: str | None = None
        best_hsv_sim: float | None = None
        best_quota_penalty = 0.0

        for idx in candidate_indices:
            if idx in selected_indices:
                continue

            cand = detections[idx]
            cand_meta = trace_details[idx]
            if max_per_source_file is not None and max_per_source_file >= 1:
                src_key = cand_meta["source_file"] or cand_meta["clip_id"]
                src_count = sum(
                    1
                    for s_idx in selected_indices
                    if (trace_details[s_idx]["source_file"] or trace_details[s_idx]["clip_id"])
                    == src_key
                )
                if src_count >= max_per_source_file:
                    continue
            r = cand_meta["relevance_score"]

            if not selected_indices:
                max_sim = 0.0
                similar_id = None
                max_hsv_sim = None
            else:
                max_sim = 0.0
                similar_id = None
                max_hsv_sim = None
                for s_idx in selected_indices:
                    s_det = detections[s_idx]
                    sim = calculate_pairwise_similarity(cand, s_det)
                    if sim > max_sim:
                        max_sim = sim
                        similar_id = trace_details[s_idx]["detection_id"]
                    hist_sim = _histogram_similarity(cand, s_det)
                    if hist_sim is not None and (max_hsv_sim is None or hist_sim > max_hsv_sim):
                        max_hsv_sim = hist_sim

            clip_id = cand_meta["clip_id"]
            count_selected_for_clip = sum(
                1 for s_idx in selected_indices if trace_details[s_idx]["clip_id"] == clip_id
            )
            q_penalty = (
                min(1.0, 0.35 * (count_selected_for_clip - per_clip_limit + 1))
                if count_selected_for_clip >= per_clip_limit
                else 0.0
            )

            mmr = round(lambda_quality * r - (1.0 - lambda_quality) * max_sim - q_penalty, 6)

            # Tie breaking by (-quality_score, -confidence, sort_key)
            tie_key = (
                mmr,
                cand_meta["quality_score"],
                cand_meta["confidence"],
                # sort_key descending inverted for min / ascending for tie
            )

            if best_idx is None:
                best_idx = idx
                best_score = mmr
                best_sim_penalty = max_sim
                best_similar_id = similar_id
                best_hsv_sim = max_hsv_sim
                best_quota_penalty = q_penalty
            else:
                current_best_meta = trace_details[best_idx]
                best_tie_key = (
                    best_score,
                    current_best_meta["quality_score"],
                    current_best_meta["confidence"],
                )
                if tie_key > best_tie_key or (
                    tie_key == best_tie_key
                    and cand_meta["sort_key"] < current_best_meta["sort_key"]
                ):
                    best_idx = idx
                    best_score = mmr
                    best_sim_penalty = max_sim
                    best_similar_id = similar_id
                    best_hsv_sim = max_hsv_sim
                    best_quota_penalty = q_penalty

        if best_idx is not None:
            selected_indices.append(best_idx)
            m = trace_details[best_idx]
            m["selected"] = True
            m["selection_rank"] = step
            m["mmr_score"] = best_score
            m["max_similarity_penalty"] = best_sim_penalty
            m["most_similar_selected_id"] = best_similar_id
            m["hsv_similarity"] = best_hsv_sim
            m["quota_penalty"] = best_quota_penalty
            m["decision_reason"] = (
                f"Selected rank {step} (MMR={best_score:.4f}, R={m['relevance_score']:.3f}, "
                f"sim={best_sim_penalty:.3f}, quota={best_quota_penalty:.2f})"
            )

    # Sort selected items chronologically by creation_time / GoPro chapter / sort_key
    selected_indices.sort(key=lambda idx: make_chronological_sort_key(detections[idx], clip_meta))

    final_selected: list[T] = []
    for chrono_rank, idx in enumerate(selected_indices, start=1):
        trace_details[idx]["chronological_rank"] = chrono_rank
        item = detections[idx]
        item = _set_val(item, "selection_rank", trace_details[idx]["selection_rank"])
        item = _set_val(item, "chronological_rank", chrono_rank)
        final_selected.append(item)

    if trace_path:
        cand_traces = [
            SelectionCandidateTrace(**meta)
            for meta in sorted(
                trace_details.values(),
                key=lambda m: (
                    0 if m["selected"] else 1,
                    m["selection_rank"] if m["selection_rank"] is not None else 9999,
                    -m["mmr_score"],
                    m["sort_key"],
                ),
            )
        ]
        out_trace = SelectionTraceOutput(
            top_n=top_n,
            lambda_quality=lambda_quality,
            per_clip_limit=per_clip_limit,
            total_candidates=len(detections),
            selected_count=len(final_selected),
            candidates=cand_traces,
        )
        t_path = Path(trace_path)
        t_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = t_path.with_name(f".{t_path.name}.tmp.{os.getpid()}")
        try:
            temp_path.write_text(out_trace.model_dump_json(indent=2), encoding="utf-8")
            temp_path.replace(t_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    return final_selected


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _det_id(det: Any) -> str:
    existing = str(_get_val(det, "detection_id", "") or "")
    if existing:
        return existing
    clip_id = str(_get_val(det, "clip_id", ""))
    start_time = float(_get_val(det, "start_time", 0.0))
    end_time = float(_get_val(det, "end_time", start_time))
    return make_detection_id(clip_id, start_time, end_time)


def _match_feedback_item(det: Any, feedback: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(feedback, Mapping):
        return None
    fb_items = feedback.get("feedback")
    if not isinstance(fb_items, Mapping):
        return None
    fname = str(_get_val(det, "file_name", "") or _get_val(det, "source_file", ""))
    start_time = float(_get_val(det, "start_time", 0.0))
    end_time = float(_get_val(det, "end_time", start_time))
    for item in fb_items.values():
        if not isinstance(item, Mapping):
            continue
        fb_src = Path(str(item.get("source_file", ""))).name
        if fb_src != fname:
            continue
        try:
            fb_start = float(item.get("start_time", -1.0))
            fb_end = float(item.get("end_time", -1.0))
        except TypeError, ValueError:
            continue
        if abs(fb_start - start_time) < 0.5 and abs(fb_end - end_time) < 0.5:
            return item
    return None


def filter_selection_candidates[T: (RefinedDetection, dict[str, Any])](
    detections: Sequence[T],
    clip_meta: Mapping[str, Mapping[str, Any]] | None = None,
    feedback: Mapping[str, Any] | None = None,
    *,
    outlier_days: int | None = 7,
    allow_outlier_dates: bool = False,
    date_range: tuple[str, str] | None = None,
    exclude_files: Collection[str] | None = None,
) -> tuple[list[T], dict[str, list[str]]]:
    """Filter unusable, bogus-subject, stream-mismatched, outlier-date, and rejected detections."""
    reasons: dict[str, list[str]] = defaultdict(list)
    excluded_set = {Path(str(f)).name for f in (exclude_files or ())}

    majority_spec: tuple[Any, Any] | None = None
    if clip_meta:
        spec_counts = Counter(
            (meta.get("fps"), meta.get("has_audio"))
            for meta in clip_meta.values()
            if meta and ("fps" in meta or "has_audio" in meta)
        )
        if spec_counts:
            majority_spec = spec_counts.most_common(1)[0][0]

    median_date: date | None = None
    if clip_meta and not allow_outlier_dates and outlier_days is not None:
        det_files = {
            str(_get_val(d, "file_name", "") or _get_val(d, "source_file", "")) for d in detections
        }
        clip_dates = sorted(
            parsed
            for fn in det_files
            if (parsed := _parse_iso_date(clip_meta.get(fn, {}).get("creation_time"))) is not None
        )
        if len(clip_dates) >= 3:
            median_date = clip_dates[len(clip_dates) // 2]

    range_start = _parse_iso_date(date_range[0]) if date_range and date_range[0] else None
    range_end = _parse_iso_date(date_range[1]) if date_range and date_range[1] else None

    kept: list[T] = []
    for raw_det in detections:
        fname = str(_get_val(raw_det, "file_name", "") or _get_val(raw_det, "source_file", ""))
        subj = str(_get_val(raw_det, "subject", "") or "").strip()
        role = str(_get_val(raw_det, "story_role", "") or "").strip().lower()
        did = _det_id(raw_det)

        if fname in excluded_set:
            reasons[f"excluded_file:{fname}"].append(did)
            continue
        if role == "none":
            reasons["story_role=none"].append(did)
            continue
        if subj.lower() in BOGUS_SUBJECTS:
            reasons[f"bogus_subject:{subj or '<empty>'}"].append(did)
            continue

        meta = clip_meta.get(fname, {}) if clip_meta else {}
        if majority_spec is not None and meta and ("fps" in meta or "has_audio" in meta):
            spec = (meta.get("fps"), meta.get("has_audio"))
            if spec != majority_spec:
                reasons[f"stream_mismatch:{fname}"].append(did)
                continue

        clip_date = _parse_iso_date(meta.get("creation_time")) if meta else None
        if (
            (range_start is not None or range_end is not None)
            and clip_date is not None
            and (
                (range_start is not None and clip_date < range_start)
                or (range_end is not None and clip_date > range_end)
            )
        ):
            reasons[f"outside_date_range:{fname}"].append(did)
            continue

        if (
            median_date is not None
            and clip_date is not None
            and outlier_days is not None
            and abs((clip_date - median_date).days) > outlier_days
        ):
            reasons[f"outlier_date:{fname}({clip_date.isoformat()})"].append(did)
            continue

        fb_item = _match_feedback_item(raw_det, feedback)
        det_item = raw_det
        if fb_item is not None:
            decision = str(fb_item.get("decision", "")).strip().lower()
            if decision == "reject":
                reasons[f"feedback_reject:{fname}"].append(did)
                continue
            if decision == "wrong_subject" and fb_item.get("corrected_subject"):
                corrected = str(fb_item["corrected_subject"]).strip()
                if isinstance(raw_det, dict):
                    copied = dict(raw_det)
                    copied["subject"] = corrected
                    det_item = copied  # type: ignore[assignment]
                elif isinstance(raw_det, BaseModel):
                    det_item = raw_det.model_copy(update={"subject": corrected})

        kept.append(det_item)

    return kept, dict(reasons)


def centered_trim_window(
    start_time: float, end_time: float, clip_duration_cap: float | None
) -> tuple[float, float]:
    """Center-trim a long window to ``clip_duration_cap`` snapped to a 0.5 s grid."""
    if clip_duration_cap is None or clip_duration_cap <= 0:
        return round(start_time, 2), round(end_time, 2)
    duration = end_time - start_time
    if duration <= clip_duration_cap:
        return round(start_time, 2), round(end_time, 2)
    mid = (start_time + end_time) / 2.0
    in_t = max(start_time, round((mid - clip_duration_cap / 2.0) * 2.0) / 2.0)
    out_t = min(end_time, round((in_t + clip_duration_cap) * 2.0) / 2.0)
    if out_t - in_t < clip_duration_cap:
        in_t = max(start_time, round((out_t - clip_duration_cap) * 2.0) / 2.0)
    return round(in_t, 2), round(out_t, 2)


def _compute_subject_clip_df(detections: Sequence[Any]) -> dict[str, int]:
    """Count distinct clip_ids per content token for underwater hero detections."""
    token_clips: dict[str, set[str]] = defaultdict(set)
    for det in detections:
        if _get_val(det, "domain", "") == "underwater" and (
            _get_val(det, "story_role", "") == "hero_subject"
        ):
            clip_id = str(_get_val(det, "clip_id", ""))
            for tok in tokenize_content_words(_get_val(det, "subject", "")):
                token_clips[tok].add(clip_id)
    return {tok: len(cids) for tok, cids in token_clips.items()}


def _enhanced_relevance(
    det: Any,
    token_clip_df: Mapping[str, int],
    *,
    keep_boost: bool = False,
) -> tuple[float, dict[str, float]]:
    base_r = calculate_intrinsic_relevance(
        _get_val(det, "quality_score", 0), _get_val(det, "confidence", 0.0)
    )
    rarity_bonus = 0.0
    domain = str(_get_val(det, "domain", "") or "")
    role = str(_get_val(det, "story_role", "") or "")
    if domain == "underwater" and role == "hero_subject":
        tokens = tokenize_content_words(_get_val(det, "subject", ""))
        if 1 <= len(tokens) <= 6:
            specific = [t for t in tokens if t not in GENERIC_RARITY_STOPWORDS]
            if specific:
                min_df = min(token_clip_df.get(t, 999) for t in specific)
                if min_df == 1:
                    rarity_bonus = 0.10
                elif min_df == 2:
                    rarity_bonus = 0.07
                elif min_df == 3:
                    rarity_bonus = 0.03

    orig_dur = float(_get_val(det, "end_time", 0.0)) - float(_get_val(det, "start_time", 0.0))
    if orig_dur >= 6.0:
        stability_bonus = 0.03
    elif role == "hero_subject" and orig_dur <= 3.6:
        stability_bonus = -0.02
    else:
        stability_bonus = 0.0

    keep_bonus = 0.08 if keep_boost else 0.0
    total = round(base_r + rarity_bonus + stability_bonus + keep_bonus, 6)
    return total, {
        "base_r": base_r,
        "rarity_bonus": rarity_bonus,
        "stability_bonus": stability_bonus,
        "keep_bonus": keep_bonus,
    }


def _bucket_for_detection(det: Any) -> str:
    domain = str(_get_val(det, "domain", "") or "")
    role = str(_get_val(det, "story_role", "") or "")
    template = str(_get_val(det, "template", "") or "")
    if domain == "generic" or role == "entry_transition":
        return "opening_context"
    if domain == "underwater" and role in {"establishing", "human_moment"}:
        return "underwater_establishing"
    if domain == "underwater" and role == "hero_subject":
        return "hero_macro" if template == "macro" else "hero_broad"
    return "hero_broad"


def _is_macro_subject(det: Any) -> bool:
    if str(_get_val(det, "template", "") or "") == "macro":
        return True
    subj_l = str(_get_val(det, "subject", "") or "").lower()
    return any(kw in subj_l for kw in MACRO_SUBJECT_KEYWORDS)


def _is_ascent_subject(det: Any) -> bool:
    subj_l = str(_get_val(det, "subject", "") or "").lower()
    return any(kw in subj_l for kw in ASCENT_SUBJECT_KEYWORDS)


def _assign_story_act(det: Any, closing_generic_id: str | None) -> int:
    did = _det_id(det)
    domain = str(_get_val(det, "domain", "") or "")
    role = str(_get_val(det, "story_role", "") or "")
    subj_l = str(_get_val(det, "subject", "") or "").lower()
    if closing_generic_id is not None and did == closing_generic_id:
        return 5
    if domain == "generic" or (role == "entry_transition" and not _is_ascent_subject(det)):
        return 1
    if _is_ascent_subject(det):
        return 5
    if role in {"establishing", "human_moment"}:
        return 4 if "cave" in subj_l else 2
    if _is_macro_subject(det):
        return 3
    return 4


def select_duration_budget_highlights(
    detections: Sequence[RefinedDetection | dict[str, Any]],
    *,
    clip_meta: Mapping[str, Mapping[str, Any]] | None = None,
    feedback: Mapping[str, Any] | None = None,
    target_duration: float = 150.0,
    clip_duration_cap: float = 6.5,
    lambda_quality: float = 0.60,
    max_per_clip: int = 1,
    max_per_source_file: int | None = 1,
    order: Literal["chronological", "story_arc"] = "story_arc",
    preset: str = "highlight",
    plan_id: str | None = None,
    title: str | None = None,
    outlier_days: int | None = 7,
    allow_outlier_dates: bool = False,
    date_range: tuple[str, str] | None = None,
    exclude_files: Collection[str] | None = None,
) -> tuple[EditPlan, dict[str, list[str]]]:
    """Select diverse highlights up to ``target_duration`` and build an executable ``EditPlan``."""
    filtered, filter_report = filter_selection_candidates(
        detections,
        clip_meta=clip_meta,
        feedback=feedback,
        outlier_days=outlier_days,
        allow_outlier_dates=allow_outlier_dates,
        date_range=date_range,
        exclude_files=exclude_files,
    )
    resolved_id = plan_id or (
        f"deterministic_{preset}_{round(target_duration)}s"
        if preset != "chronological"
        else f"deterministic_chrono_{round(target_duration)}s"
    )
    resolved_title = (
        title or f"Deterministic {preset.replace('_', ' ').title()} ({round(target_duration)}s)"
    )

    if not filtered or target_duration <= 0:
        empty_plan = EditPlan(
            id=resolved_id,
            title=resolved_title,
            style=order,
            preset=preset,
            target_duration_seconds=max(1, round(target_duration)),
            estimated_duration_seconds=0.0,
            clip_count=0,
            narrative_summary="No eligible detections available after filtering.",
            acts=[],
            shots=[],
        )
        return empty_plan, filter_report

    effective_cap = clip_duration_cap
    if preset == "teaser":
        effective_cap = min(clip_duration_cap, 3.5)
        bucket_weights = [
            ("opening_context", 0.15),
            ("underwater_establishing", 0.10),
            ("hero_macro", 0.35),
            ("hero_broad", 0.40),
        ]
    else:
        bucket_weights = [
            ("opening_context", 0.20),
            ("underwater_establishing", 0.20),
            ("hero_macro", 0.25),
            ("hero_broad", 0.35),
        ]

    token_df = _compute_subject_clip_df(filtered)
    entries: list[dict[str, Any]] = []
    for det in filtered:
        did = _det_id(det)
        start_time = float(_get_val(det, "start_time", 0.0))
        end_time = float(_get_val(det, "end_time", start_time))
        in_t, out_t = centered_trim_window(start_time, end_time, effective_cap)
        if out_t <= in_t:
            continue
        fb = _match_feedback_item(det, feedback)
        keep_boost = bool(fb and str(fb.get("decision", "")).strip().lower() == "keep")
        rel, rel_parts = _enhanced_relevance(det, token_df, keep_boost=keep_boost)
        entries.append({
            "det": det,
            "detection_id": did,
            "clip_id": str(_get_val(det, "clip_id", "")),
            "file_name": str(_get_val(det, "file_name", "") or _get_val(det, "source_file", "")),
            "file_path": str(_get_val(det, "file_path", "")),
            "in_time": in_t,
            "out_time": out_t,
            "trimmed_duration": round(out_t - in_t, 2),
            "enhanced_relevance": rel,
            "relevance_parts": rel_parts,
            "bucket": _bucket_for_detection(det),
        })

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    clip_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total_dur = 0.0

    def _pick_into_bucket(pool: list[dict[str, Any]], budget: float) -> None:
        nonlocal total_dur
        bucket_dur = 0.0
        while bucket_dur < budget and total_dur < target_duration - 1.5:
            best_entry: dict[str, Any] | None = None
            best_mmr = -float("inf")
            for entry in pool:
                did = entry["detection_id"]
                if did in selected_ids:
                    continue
                src_key = entry["file_name"] or entry["clip_id"]
                if (
                    max_per_source_file is not None
                    and max_per_source_file >= 1
                    and source_counts[src_key] >= max_per_source_file
                ):
                    continue
                cdur = float(entry["trimmed_duration"])
                if selected and total_dur + cdur > target_duration + 4.0:
                    continue

                cand = entry["det"]
                max_sim = (
                    max(calculate_pairwise_similarity(cand, s["det"]) for s in selected)
                    if selected
                    else 0.0
                )
                cnt = clip_counts[entry["clip_id"]]
                q_pen = min(1.0, 0.35 * (cnt - max_per_clip + 1)) if cnt >= max_per_clip else 0.0
                mmr = round(
                    lambda_quality * entry["enhanced_relevance"]
                    - (1.0 - lambda_quality) * max_sim
                    - q_pen,
                    6,
                )
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_entry = entry

            if best_entry is None:
                break
            chosen = dict(best_entry)
            chosen["selection_rank"] = len(selected) + 1
            chosen["mmr_score"] = best_mmr
            selected.append(chosen)
            selected_ids.add(chosen["detection_id"])
            clip_counts[chosen["clip_id"]] += 1
            source_counts[chosen["file_name"] or chosen["clip_id"]] += 1
            bucket_dur += float(chosen["trimmed_duration"])
            total_dur += float(chosen["trimmed_duration"])

    for bucket_name, weight in bucket_weights:
        _pick_into_bucket(
            [e for e in entries if e["bucket"] == bucket_name], target_duration * weight
        )

    if total_dur < target_duration - 1.5:
        _pick_into_bucket(
            [e for e in entries if e["bucket"] in {"hero_macro", "hero_broad"}],
            target_duration - total_dur,
        )
    if total_dur < target_duration - 1.5:
        _pick_into_bucket(entries, target_duration - total_dur)

    if order == "chronological":
        ordered = sorted(selected, key=lambda e: make_chronological_sort_key(e["det"], clip_meta))
        for entry in ordered:
            entry["act"] = 1
            entry["act_title"] = "Chronological Sequence"
    else:
        generic_entries = [
            e for e in selected if str(_get_val(e["det"], "domain", "")) == "generic"
        ]
        closing_generic_id: str | None = None
        if len(generic_entries) >= 2:
            generic_sorted = sorted(
                generic_entries, key=lambda e: make_chronological_sort_key(e["det"], clip_meta)
            )
            closing_generic_id = generic_sorted[-1]["detection_id"]
        for entry in selected:
            act_num = _assign_story_act(entry["det"], closing_generic_id)
            entry["act"] = act_num
            entry["act_title"] = ACT_TITLES.get(act_num, f"Act {act_num}")
        ordered = sorted(
            selected,
            key=lambda e: (int(e["act"]), make_chronological_sort_key(e["det"], clip_meta)),
        )

    shots: list[EditShot] = []
    acts_map: dict[int, list[EditShot]] = defaultdict(list)
    total_runtime = 0.0
    n_shots = len(ordered)
    for idx, entry in enumerate(ordered):
        det = entry["det"]
        act_num = int(entry["act"])
        act_title = str(entry["act_title"])
        prev_act = int(ordered[idx - 1]["act"]) if idx > 0 else None
        next_act = int(ordered[idx + 1]["act"]) if idx < n_shots - 1 else None
        fade_in = idx == 0 or (order == "story_arc" and prev_act != act_num)
        fade_out = idx == n_shots - 1 or (order == "story_arc" and next_act != act_num)

        fname = entry["file_name"]
        ctime = ""
        if clip_meta and fname in clip_meta:
            ctime = str(clip_meta[fname].get("creation_time") or "")
        parts = entry["relevance_parts"]
        rationale = (
            f"MMR rank #{entry['selection_rank']} (score={entry['mmr_score']:.3f}, "
            f"rarity=+{parts['rarity_bonus']:.2f}, stab={parts['stability_bonus']:+.2f})"
        )
        shot = EditShot(
            detection_id=entry["detection_id"],
            clip_id=entry["clip_id"],
            file_name=fname,
            file_path=entry["file_path"],
            creation_time=ctime,
            act=act_num,
            act_title=act_title,
            in_time=entry["in_time"],
            out_time=entry["out_time"],
            duration=entry["trimmed_duration"],
            subject=str(_get_val(det, "subject", "") or ""),
            story_role=str(_get_val(det, "story_role", "hero_subject") or "hero_subject"),
            quality_score=int(_get_val(det, "quality_score", 0) or 0),
            transition_in="fade" if fade_in else "cut",
            transition_out="fade" if fade_out else "cut",
            rationale=rationale,
        )
        shots.append(shot)
        acts_map[act_num].append(shot)
        total_runtime += shot.duration

    acts = [
        EditAct(
            act_number=act_num,
            title=act_shots[0].act_title,
            goal=(
                "Chronological highlight sequence"
                if order == "chronological"
                else "Role-balanced narrative beat"
            ),
            shots=act_shots,
        )
        for act_num, act_shots in sorted(acts_map.items())
    ]

    plan = EditPlan(
        id=resolved_id,
        title=resolved_title,
        style=order,
        preset=preset,
        target_duration_seconds=max(1, round(target_duration)),
        estimated_duration_seconds=round(total_runtime, 2),
        clip_count=len(shots),
        narrative_summary=(
            f"Deterministic {order} selection targeting {round(target_duration)}s "
            f"(max {max_per_source_file or 'unlimited'} per source file)."
        ),
        acts=acts,
        shots=shots,
    )
    return plan, filter_report
