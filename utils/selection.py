"""Maximal Marginal Relevance (MMR) highlight selection and shot tokenization."""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from utils.models import RefinedDetection, SelectionCandidateTrace, SelectionTraceOutput

logger = logging.getLogger(__name__)

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


def calculate_pairwise_similarity(d_i: Any, d_j: Any) -> float:
    """Calculate S(d_i, d_j) in [0.0, 1.0] across subject, description, category, and time."""
    subj_i = str(_get_val(d_i, "subject", "") or "").strip().lower()
    subj_j = str(_get_val(d_j, "subject", "") or "").strip().lower()

    if subj_i and subj_i == subj_j:
        j_subj = 1.0
    else:
        j_subj = _jaccard(tokenize_content_words(subj_i), tokenize_content_words(subj_j))

    desc_i = str(_get_val(d_i, "description", "") or "")
    desc_j = str(_get_val(d_j, "description", "") or "")
    j_desc = _jaccard(tokenize_content_words(desc_i), tokenize_content_words(desc_j))

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
    return max(0.0, min(1.0, similarity))


def make_detection_id(clip_id: str, start_time: float, end_time: float) -> str:
    """Build deterministic detection identifier."""
    return f"{clip_id}_{start_time:08.2f}_{end_time:08.2f}"


def make_sort_key(source_file: str, start_time: float, end_time: float, clip_id: str) -> str:
    """Build canonical chronological sort key."""
    return f"{source_file}:{start_time:08.2f}:{end_time:08.2f}:{clip_id}"


def select_diverse_highlights[T: (RefinedDetection, dict[str, Any])](
    detections: list[T],
    top_n: int,
    lambda_quality: float = 0.65,
    max_per_clip: int | None = None,
    trace_path: Path | str | None = None,
) -> list[T]:
    """Select up to top_n diverse highlights using Maximal Marginal Relevance (MMR).

    Returns selected detections sorted chronologically by (source_file, start_time, end_time, clip_id).
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
        }

    k_target = min(top_n, len(detections))

    for step in range(1, k_target + 1):
        best_idx: int | None = None
        best_score = -float("inf")
        best_sim_penalty = 0.0
        best_similar_id: str | None = None
        best_quota_penalty = 0.0

        for idx in candidate_indices:
            if idx in selected_indices:
                continue

            cand = detections[idx]
            cand_meta = trace_details[idx]
            r = cand_meta["relevance_score"]

            if not selected_indices:
                max_sim = 0.0
                similar_id = None
            else:
                max_sim = 0.0
                similar_id = None
                for s_idx in selected_indices:
                    s_det = detections[s_idx]
                    sim = calculate_pairwise_similarity(cand, s_det)
                    if sim > max_sim:
                        max_sim = sim
                        similar_id = trace_details[s_idx]["detection_id"]

            clip_id = cand_meta["clip_id"]
            count_selected_for_clip = sum(
                1 for s_idx in selected_indices if trace_details[s_idx]["clip_id"] == clip_id
            )
            q_penalty = 0.35 if count_selected_for_clip >= per_clip_limit else 0.0

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
                    best_quota_penalty = q_penalty

        if best_idx is not None:
            selected_indices.append(best_idx)
            m = trace_details[best_idx]
            m["selected"] = True
            m["selection_rank"] = step
            m["mmr_score"] = best_score
            m["max_similarity_penalty"] = best_sim_penalty
            m["most_similar_selected_id"] = best_similar_id
            m["quota_penalty"] = best_quota_penalty
            m["decision_reason"] = (
                f"Selected rank {step} (MMR={best_score:.4f}, R={m['relevance_score']:.3f}, "
                f"sim={best_sim_penalty:.3f}, quota={best_quota_penalty:.2f})"
            )

    # Sort selected items chronologically by sort_key
    selected_indices.sort(key=lambda idx: trace_details[idx]["sort_key"])

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
        temp_path.write_text(out_trace.model_dump_json(indent=2), encoding="utf-8")
        temp_path.replace(t_path)

    return final_selected
