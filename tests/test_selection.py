"""Tests for canonical MMR highlight selection and tokenization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.models import RefinedDetection
from utils.selection import (
    STOPWORDS,
    calculate_intrinsic_relevance,
    calculate_pairwise_similarity,
    select_diverse_highlights,
    tokenize_content_words,
)


def test_tokenize_content_words_filtering_and_determinism() -> None:
    """tokenize_content_words should remove stopwords, filter short tokens, and be deterministic."""
    text1 = "The green sea turtle was swimming quickly into clear water!"
    tokens1 = tokenize_content_words(text1)
    assert "the" not in tokens1
    assert "sea" in tokens1
    assert "turtle" in tokens1
    assert "quickly" in tokens1
    assert "swimming" not in tokens1
    assert "clear" not in tokens1
    assert "water" not in tokens1

    # Short tokens (< 3 chars) excluded
    assert tokenize_content_words("a in on go 42") == frozenset()

    # Empty/None input
    assert tokenize_content_words("") == frozenset()
    assert tokenize_content_words(None) == frozenset()  # type: ignore[arg-type]


def test_stopwords_contain_expected_terms() -> None:
    """STOPWORDS contains specified domain-neutral filler words."""
    for word in ("the", "and", "swimming", "riding", "moving", "water", "road", "scene"):
        assert word in STOPWORDS


def test_relevance_score_formula() -> None:
    """R(d) = 0.85 * (quality / 10.0) + 0.15 * confidence."""
    # quality=10, confidence=1.0 -> 1.0
    r_max = calculate_intrinsic_relevance(quality_score=10, confidence=1.0)
    assert pytest.approx(r_max, rel=1e-5) == 1.0

    # quality=5, confidence=0.8 -> 0.85 * 0.5 + 0.15 * 0.8 = 0.425 + 0.120 = 0.545
    r_mid = calculate_intrinsic_relevance(quality_score=5, confidence=0.8)
    assert pytest.approx(r_mid, rel=1e-5) == 0.545

    # Clamping: quality 0 clamped to 1, quality 12 clamped to 10
    assert (
        pytest.approx(calculate_intrinsic_relevance(quality_score=0, confidence=0.0), rel=1e-5)
        == 0.085
    )
    assert (
        pytest.approx(calculate_intrinsic_relevance(quality_score=12, confidence=1.5), rel=1e-5)
        == 1.0
    )


def test_pairwise_similarity_components() -> None:
    """Pairwise similarity combines subject, description, category, and temporal proximity."""
    d1 = {
        "clip_id": "clip_1",
        "start_time": 10.0,
        "end_time": 15.0,
        "subject": "green sea turtle",
        "description": "turtle swimming over shallow reef",
        "template": "pelagics",
        "story_role": "hero_subject",
    }
    d2 = {
        "clip_id": "clip_1",
        "start_time": 20.0,
        "end_time": 25.0,
        "subject": "sea turtle swimming",
        "description": "giant turtle hovering near coral reef",
        "template": "pelagics",
        "story_role": "hero_subject",
    }
    # Both are clip_1. Midpoint diff: 22.5 - 12.5 = 10.0s. T_prox = 1.0 - 10.0/60.0 = 50.0/60.0 ≈ 0.8333
    # Same category/template: I_cat = 1.0
    sim = calculate_pairwise_similarity(d1, d2)
    assert 0.0 < sim <= 1.0

    # Different clip_id: T_prox must be 0.0
    d3 = dict(d2, clip_id="clip_2")
    sim_diff_clip = calculate_pairwise_similarity(d1, d3)
    assert sim_diff_clip < sim


def test_select_diverse_highlights_penalizes_synonyms_and_monopoly(tmp_path: Path) -> None:
    """MMR should select diverse subjects across clips rather than redundant windows of one clip."""
    # Clip 1 has 3 consecutive high-quality turtle detections (quality 10, 9, 9)
    # Clip 2 has a high-quality manta ray detection (quality 8)
    # Clip 3 has a high-quality clownfish detection (quality 8)
    detections = [
        RefinedDetection(
            clip_id="clip_1",
            file_name="clip_1.mp4",
            file_path="/videos/clip_1.mp4",
            start_time=10.0,
            end_time=15.0,
            subject="green sea turtle",
            confidence=0.95,
            quality_score=10,
            description="green turtle eating grass",
            template="pelagics",
        ),
        RefinedDetection(
            clip_id="clip_1",
            file_name="clip_1.mp4",
            file_path="/videos/clip_1.mp4",
            start_time=20.0,
            end_time=25.0,
            subject="sea turtle swimming",
            confidence=0.92,
            quality_score=9,
            description="sea turtle cruising past camera",
            template="pelagics",
        ),
        RefinedDetection(
            clip_id="clip_1",
            file_name="clip_1.mp4",
            file_path="/videos/clip_1.mp4",
            start_time=35.0,
            end_time=40.0,
            subject="hawksbill turtle",
            confidence=0.90,
            quality_score=9,
            description="hawksbill turtle resting",
            template="pelagics",
        ),
        RefinedDetection(
            clip_id="clip_2",
            file_name="clip_2.mp4",
            file_path="/videos/clip_2.mp4",
            start_time=5.0,
            end_time=10.0,
            subject="manta ray",
            confidence=0.90,
            quality_score=8,
            description="manta ray barrel rolling",
            template="pelagics",
        ),
        RefinedDetection(
            clip_id="clip_3",
            file_name="clip_3.mp4",
            file_path="/videos/clip_3.mp4",
            start_time=12.0,
            end_time=17.0,
            subject="clownfish anemone",
            confidence=0.88,
            quality_score=8,
            description="clownfish guarding anemone home",
            template="macro",
        ),
    ]

    trace_file = tmp_path / "selection_trace.json"
    selected = select_diverse_highlights(
        detections,
        top_n=3,
        lambda_quality=0.65,
        trace_path=trace_file,
    )

    assert len(selected) == 3

    # The selected items should include diversity (e.g. manta ray or clownfish, not just 3 turtles)
    selected_subjects = [d.subject for d in selected]
    assert any("turtle" in s for s in selected_subjects)
    assert any("manta" in s for s in selected_subjects)
    assert any("clownfish" in s for s in selected_subjects)

    # Chronological sort order by (source_file, start_time, end_time, clip_id)
    sort_keys = [(d.file_name, d.start_time) for d in selected]
    assert sort_keys == sorted(sort_keys)

    # Trace file check
    assert trace_file.exists()
    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    assert trace_data["top_n"] == 3
    assert trace_data["selected_count"] == 3
    assert len(trace_data["candidates"]) == 5

    # Check candidates in trace have required keys
    for cand in trace_data["candidates"]:
        assert "detection_id" in cand
        assert "sort_key" in cand
        assert "mmr_score" in cand
        assert "decision_reason" in cand
        if cand["selected"]:
            assert cand["selection_rank"] is not None
            assert cand["chronological_rank"] is not None


def test_select_diverse_highlights_dict_input() -> None:
    """select_diverse_highlights should support dict-based detection lists."""
    detections = [
        {
            "clip_id": "clip_a",
            "file_name": "video_a.mp4",
            "start_time": 5.0,
            "end_time": 10.0,
            "subject": "shark",
            "quality_score": 9,
            "confidence": 0.9,
            "description": "reef shark",
        },
        {
            "clip_id": "clip_b",
            "file_name": "video_b.mp4",
            "start_time": 15.0,
            "end_time": 20.0,
            "subject": "diver",
            "quality_score": 8,
            "confidence": 0.85,
            "description": "diver exploring",
        },
    ]

    selected = select_diverse_highlights(detections, top_n=2)
    assert len(selected) == 2
    assert isinstance(selected[0], dict)
    assert selected[0]["selection_rank"] is not None
    assert selected[0]["chronological_rank"] is not None
