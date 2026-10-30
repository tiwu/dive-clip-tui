"""Tests for scripts/04_plan_storyline.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load scripts/04_plan_storyline.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_storyline_spec = importlib.util.spec_from_file_location(
    "scripts_04_plan_storyline", _SCRIPTS_DIR / "04_plan_storyline.py"
)
storyline = importlib.util.module_from_spec(_storyline_spec)
_storyline_spec.loader.exec_module(storyline)


def test_build_prompt() -> None:
    """build_prompt should create a prompt with detections and target duration."""
    from utils.models import RefinedDetection, DetectionsRefined

    det1 = RefinedDetection(
        clip_id="clip001",
        file_name="dive1.mp4",
        file_path="/tmp/dive1.mp4",
        fps="25.0",
        start_time=0.0,
        end_time=5.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="Sea turtle swimming",
        color="Cyan",
        template="pelagics",
    )
    detections = DetectionsRefined(
        total_clips=1, total_detections=1, detections=[det1]
    )

    prompt = storyline.build_prompt(detections, target_duration=4)

    assert "professional underwater video editor" in prompt
    assert "4 minutes" in prompt
    assert "dive1.mp4" in prompt
    assert "turtle" in prompt
    assert "8/10" in prompt


def test_build_prompt_empty_detections() -> None:
    """build_prompt should handle empty detections list."""
    from utils.models import DetectionsRefined

    detections = DetectionsRefined(total_clips=0, total_detections=0, detections=[])

    prompt = storyline.build_prompt(detections, target_duration=4)

    assert "professional underwater video editor" in prompt
    assert "4 minutes" in prompt
