"""Tests for scripts/02_scan_footage.py."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from utils import ollama_client
from utils.prompt_templates import (
    BROAD_MARINE_TEMPLATE,
    MACRO_TEMPLATE,
    PELAGICS_TEMPLATE,
)

# Load scripts/02_scan_footage.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_scan_spec = importlib.util.spec_from_file_location(
    "scripts_02_scan_footage", _SCRIPTS_DIR / "02_scan_footage.py"
)
scan = importlib.util.module_from_spec(_scan_spec)
_scan_spec.loader.exec_module(scan)


def test_deduplicate_detections_empty() -> None:
    """_deduplicate_detections should return empty list for empty input."""
    assert scan._deduplicate_detections([]) == []


def test_deduplicate_detections_no_overlap() -> None:
    """_deduplicate_detections should keep all non-overlapping detections."""
    from utils.models import RefinedDetection

    det1 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=0.0,
        end_time=5.0,
        subject="a",
        confidence=0.8,
        quality_score=8,
        description="a",
    )
    det2 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=10.0,
        end_time=15.0,
        subject="b",
        confidence=0.9,
        quality_score=9,
        description="b",
    )
    result = scan._deduplicate_detections([det1, det2])
    assert len(result) == 2


def test_deduplicate_detections_overlap_keeps_higher_quality() -> None:
    """_deduplicate_detections should keep higher quality score when overlapping."""
    from utils.models import RefinedDetection

    det_low = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=0.0,
        end_time=10.0,
        subject="a",
        confidence=0.8,
        quality_score=5,
        description="low",
    )
    det_high = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=5.0,
        end_time=15.0,
        subject="b",
        confidence=0.9,
        quality_score=9,
        description="high",
    )
    result = scan._deduplicate_detections([det_low, det_high])
    assert len(result) == 1
    assert result[0].quality_score == 9


def test_deduplicate_detections_merges_contiguous_same_subject() -> None:
    """_deduplicate_detections should merge contiguous windows of the same subject into one continuous highlight."""
    from utils.models import RefinedDetection

    det1 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=10.0,
        end_time=12.0,
        subject="turtle",
        confidence=0.85,
        quality_score=8,
        description="turtle swimming",
    )
    det2 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=12.0,
        end_time=14.0,
        subject="turtle",
        confidence=0.95,
        quality_score=9,
        description="turtle close up",
    )
    result = scan._deduplicate_detections([det1, det2])
    assert len(result) == 1
    assert result[0].start_time == 10.0
    assert result[0].end_time == 14.0
    assert result[0].quality_score == 9
    assert result[0].confidence == 0.95
    assert result[0].description == "turtle close up"


def test_deduplicate_detections_merges_fuzzy_token_overlap() -> None:
    """_deduplicate_detections should merge contiguous windows sharing content tokens."""
    from utils.models import MergeEvent, RefinedDetection

    det1 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=10.0,
        end_time=12.0,
        subject="green sea turtle",
        confidence=0.85,
        quality_score=8,
        description="turtle swimming",
    )
    det2 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=12.5,
        end_time=14.5,
        subject="sea turtle swimming",
        confidence=0.95,
        quality_score=9,
        description="turtle close up",
    )
    merge_events: list[MergeEvent] = []
    result = scan._deduplicate_detections([det1, det2], trace_merge_events=merge_events)
    assert len(result) == 1
    assert result[0].start_time == 10.0
    assert result[0].end_time == 14.5
    assert result[0].quality_score == 9
    assert result[0].subject == "sea turtle swimming"
    assert len(merge_events) == 1
    assert merge_events[0].action == "merged"
    assert "turtle" in merge_events[0].matched_tokens


def test_deduplicate_detections_splits_on_max_clip_duration() -> None:
    """_deduplicate_detections should split into separate segments when max_clip_duration is exceeded."""
    from utils.models import MergeEvent, RefinedDetection

    det1 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=10.0,
        end_time=20.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="turtle moving",
    )
    det2 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=20.5,
        end_time=24.0,  # 24.0 - 10.0 = 14.0s > 12.0s
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="turtle continuing",
    )
    merge_events: list[MergeEvent] = []
    result = scan._deduplicate_detections(
        [det1, det2],
        max_clip_duration=12.0,
        trace_merge_events=merge_events,
    )
    assert len(result) == 2
    assert result[0].start_time == 10.0
    assert result[0].end_time == 20.0
    assert result[1].start_time == 20.5
    assert result[1].end_time == 24.0
    assert any(ev.action == "split_max_duration" for ev in merge_events)


def test_deduplicate_detections_preserves_single_char_unrelated_subjects() -> None:
    """Single character subjects ('a' vs 'b') should not merge."""
    from utils.models import RefinedDetection

    det1 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=10.0,
        end_time=12.0,
        subject="a",
        confidence=0.85,
        quality_score=8,
        description="test a",
    )
    det2 = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=11.0,
        end_time=13.0,
        subject="b",
        confidence=0.95,
        quality_score=9,
        description="test b",
    )
    result = scan._deduplicate_detections([det1, det2])
    # Overlapping windows with different single-letter subjects: Step 2 keeps higher quality
    assert len(result) == 1
    assert result[0].subject == "b"


def test_normalize_vlm_frame_response() -> None:
    """normalize_vlm_frame_response sanitizes invalid story_role and strips arbitrary keys."""
    raw = {
        "detected": True,
        "subject": "green turtle",
        "category": "pelagics",
        "story_role": "invalid_role",
        "confidence": 1.5,
        "quality_score": 11,
        "reject_reasons": "some string reason",
        "description": "good shot",
        "timestamp": 12.5,
        "invented_vlm_key": "ignore me",
    }
    norm = scan.normalize_vlm_frame_response(raw)
    assert norm["detected"] is True
    assert norm["subject"] == "green turtle"
    assert norm["category"] == "pelagics"
    # Invalid role with detected=True defaults to hero_subject
    assert norm["story_role"] == "hero_subject"
    # Clamping
    assert norm["confidence"] == 1.0
    assert norm["quality_score"] == 10
    assert norm["reject_reasons"] == ["some string reason"]
    assert "invented_vlm_key" not in norm

    # False detected defaults story_role to none
    raw_false = {
        "detected": False,
        "subject": "none",
        "category": "none",
        "confidence": -0.5,
        "quality_score": None,
    }
    norm_false = scan.normalize_vlm_frame_response(raw_false)
    assert norm_false["detected"] is False
    assert norm_false["story_role"] == "none"
    assert norm_false["confidence"] == 0.0
    assert norm_false["quality_score"] == 0
    assert norm_false["reject_reasons"] == []

    # Defensive recovery: detected=False but valid subject and no reject reasons
    raw_recovery = {
        "detected": False,
        "subject": "stonefish",
        "category": "macro",
        "story_role": "hero_subject",
        "confidence": 0.95,
        "quality_score": 8,
        "reject_reasons": [],
        "description": "A stonefish resting on coral",
    }
    norm_rec = scan.normalize_vlm_frame_response(raw_recovery)
    assert norm_rec["detected"] is True
    assert norm_rec["subject"] == "stonefish"
    assert norm_rec["confidence"] == 0.95
    assert norm_rec["quality_score"] == 8
    assert norm_rec["reject_reasons"] == []


def test_call_ollama_success(tmp_path: Path) -> None:
    """call_ollama should parse Ollama JSON response correctly."""
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": '{"detected": true, "subject": "turtle"}'}

    with patch.object(ollama_client.requests, "post", return_value=mock_response):
        result = scan.call_ollama(
            "http://localhost:11434", "qwen2.5-vl:7b", str(image_path), "prompt"
        )

    assert result is not None
    assert result["detected"] is True
    assert result["subject"] == "turtle"


def test_call_ollama_forwards_system_prompt(tmp_path: Path) -> None:
    """call_ollama forwards system_prompt to OllamaClient."""
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"data")
    mock_client = MagicMock()
    mock_client.generate.return_value = {"detected": True}

    scan.call_ollama(
        "http://localhost:11434",
        "model",
        str(image_path),
        "prompt",
        client=mock_client,
        system_prompt="Custom system instructions",
    )
    mock_client.generate.assert_called_once_with(
        str(image_path),
        "prompt",
        "unspecified",
        system_prompt="Custom system instructions",
    )


def test_call_ollama_request_failure(tmp_path: Path) -> None:
    """call_ollama should return None when request fails."""
    import requests

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")

    with patch.object(
        ollama_client.requests,
        "post",
        side_effect=requests.exceptions.RequestException("connection error"),
    ):
        result = scan.call_ollama(
            "http://localhost:11434", "qwen2.5-vl:7b", str(image_path), "prompt"
        )

    assert result is None


def test_call_ollama_invalid_json(tmp_path: Path) -> None:
    """call_ollama should return None when response is invalid JSON."""
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "not json"}

    with patch.object(ollama_client.requests, "post", return_value=mock_response):
        result = scan.call_ollama(
            "http://localhost:11434", "qwen2.5-vl:7b", str(image_path), "prompt"
        )

    assert result is None


def test_process_clip_no_candidates(tmp_path: Path) -> None:
    """process_clip should return empty list when no candidates found."""
    clip = {
        "file_path": str(tmp_path / "video.mp4"),
        "file_name": "video.mp4",
        "clip_id": "c1",
        "fps": "25.0",
        "tags": [],
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    # Mock extract_frames to return no candidates
    with patch.object(scan, "extract_frames", return_value=[]):
        result = scan.process_clip(clip, config, tmp_path / "work")

    assert result == []


def test_process_clip_with_candidates(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """process_clip should run coarse-to-fine scan and return detections."""
    caplog.set_level(logging.DEBUG)
    video = tmp_path / "video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    coarse_frames = [
        {"frame_path": str(tmp_path / "frame_000001.jpg"), "timestamp": 2.0, "frame_number": 1},
    ]
    (tmp_path / "frame_000001.jpg").touch()

    fine_frames = [
        {
            "frame_path": str(tmp_path / "fine_frame_000001.jpg"),
            "timestamp": 3.0,
            "frame_number": 1,
        },
    ]
    (tmp_path / "fine_frame_000001.jpg").touch()

    def mock_extract_frames(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if start_seconds is not None:
            (output_dir / "fine_frame_000001.jpg").touch()
            return fine_frames
        (output_dir / "frame_000001.jpg").touch()
        return coarse_frames

    mock_ollama_result = {
        "detected": True,
        "subject": "turtle",
        "confidence": 0.8,
        "quality_score": 8,
        "description": "Sea turtle",
    }

    response = MagicMock()
    response.json.return_value = {"response": json.dumps(mock_ollama_result)}
    client = ollama_client.OllamaClient("http://localhost:11434", "qwen2.5-vl:7b", 90)
    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract_frames),
        patch.object(ollama_client.requests, "post", return_value=response),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work", client)

    assert len(result) > 0
    assert result[0].subject == "turtle"
    assert result[0].quality_score == 8
    assert result[0].clip_id.startswith("clip_")
    assert result[0].fps == "25.0"
    assert "[Pass 1] Analyzing frame 1 of 1" in caplog.text
    assert "[Pass 2] Window 1 of 1: analyzing frame 1 of 1" in caplog.text
    # Frame-matching observability: thresholds must be tunable from the log alone.
    assert "[Pass 1] Summary video.mp4" in caplog.text
    assert "confidence min/avg/max" in caplog.text
    assert (
        "[Pass 1] Frame 1 @ 2.0s: detected=True confidence=0.80 quality=8 passes_threshold=True"
        in caplog.text
    )
    assert "[Pass 2] Window 1 of 1: 0.0s-" in caplog.text
    assert "[Pass 2] Window 1 of 1: extracted 1 frames" in caplog.text
    assert "detected=True" in caplog.text and "quality=" in caplog.text
    assert "[Pass 2] Summary video.mp4" in caplog.text
    assert "after dedup" in caplog.text
    assert client.metrics.by_phase["coarse"].calls == 1
    assert client.metrics.by_phase["fine"].calls == 1


def test_fallback_uses_broad_marine_in_coarse_pass(tmp_path: Path) -> None:
    """When tags don't match any template, coarse pass should use broad marine template."""
    video = tmp_path / "video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "tags": [],
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    # Mock extract_frames to return coarse frames so call_ollama is invoked
    coarse_frame_path = tmp_path / "frame_000001.jpg"
    coarse_frame_path.touch()

    def mock_extract_frames_coarse(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_000001.jpg").touch()
        return [
            {"frame_path": str(coarse_frame_path), "timestamp": 2.0, "frame_number": 1},
        ]

    captured_prompts: list[str] = []

    def mock_call_ollama(*args, **kwargs):
        # call_ollama signature: call_ollama(host, model, image_path, prompt, timeout, client=None, phase="coarse")
        prompt = args[3] if len(args) > 3 else kwargs.get("prompt", "")
        captured_prompts.append(prompt)
        return {
            "detected": True,
            "subject": "fish",
            "confidence": 0.8,
            "quality_score": 8,
            "description": "Fish seen",
            "timestamp": 2.0,
        }

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract_frames_coarse),
        patch.object(scan, "call_ollama", side_effect=mock_call_ollama),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work")

    # Should have used broad marine template in coarse pass since tags are empty
    assert len(captured_prompts) >= 1
    assert any(BROAD_MARINE_TEMPLATE.coarse_prompt in p for p in captured_prompts), (
        f"Expected broad marine template prompt in coarse pass, got prompts: {captured_prompts}"
    )
    assert len(result) > 0


def test_fallback_uses_broad_marine_in_fine_pass(tmp_path: Path) -> None:
    """When tags don't match any template, fine pass should use broad marine template."""
    video = tmp_path / "video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "tags": [],
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    # Coarse pass: detect a window
    coarse_frame_path = tmp_path / "frame_000001.jpg"
    coarse_frame_path.touch()
    fine_frame_path = tmp_path / "fine_frame_000001.jpg"
    fine_frame_path.touch()

    def mock_extract_frames_coarse_fine(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if start_seconds is not None:
            (output_dir / "fine_frame_000001.jpg").touch()
            return [
                {"frame_path": str(fine_frame_path), "timestamp": 3.0, "frame_number": 1},
            ]
        (output_dir / "frame_000001.jpg").touch()
        return [
            {"frame_path": str(coarse_frame_path), "timestamp": 2.0, "frame_number": 1},
        ]

    captured_prompts: list[str] = []

    def mock_call_ollama(*args, **kwargs):
        prompt = args[3] if len(args) > 3 else kwargs.get("prompt", "")
        captured_prompts.append(prompt)
        return {
            "detected": True,
            "subject": "fish",
            "confidence": 0.8,
            "quality_score": 8,
            "description": "Fish seen",
            "timestamp": 2.0,
        }

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract_frames_coarse_fine),
        patch.object(scan, "call_ollama", side_effect=mock_call_ollama),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work")

    # Should have used broad marine template in fine pass since tags are empty
    assert len(captured_prompts) >= 2  # coarse + fine
    fine_prompt = captured_prompts[-1]  # Last call is fine pass
    assert BROAD_MARINE_TEMPLATE.fine_prompt in fine_prompt, (
        f"Expected broad marine template prompt in fine pass, got: {fine_prompt[:80]}..."
    )
    assert len(result) > 0


def test_matched_template_preserves_pelagics(tmp_path: Path) -> None:
    """When tags match pelagics, pelagics template should be used (existing behavior preserved)."""
    video = tmp_path / "video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "tags": ["turtle", "shark"],
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    coarse_frame_path = tmp_path / "frame_000001.jpg"
    coarse_frame_path.touch()

    def mock_extract_frames_pelagics(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_000001.jpg").touch()
        return [
            {"frame_path": str(coarse_frame_path), "timestamp": 2.0, "frame_number": 1},
        ]

    captured_prompts: list[str] = []

    def mock_call_ollama(*args, **kwargs):
        prompt = args[3] if len(args) > 3 else kwargs.get("prompt", "")
        captured_prompts.append(prompt)
        return {
            "detected": True,
            "subject": "turtle",
            "confidence": 0.8,
            "quality_score": 8,
            "description": "Sea turtle",
            "timestamp": 2.0,
        }

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract_frames_pelagics),
        patch.object(scan, "call_ollama", side_effect=mock_call_ollama),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work")

    # Should have used pelagics template since tags match
    assert len(captured_prompts) >= 1
    prompt_used = captured_prompts[0]
    assert PELAGICS_TEMPLATE.coarse_prompt in prompt_used, (
        f"Expected pelagics template prompt, got: {prompt_used[:80]}..."
    )
    assert len(result) > 0


def test_matched_template_preserves_macro(tmp_path: Path) -> None:
    """When tags match macro, macro template should be used (existing behavior preserved)."""
    video = tmp_path / "video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "tags": ["nudibranch", "coral"],
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    coarse_frame_path = tmp_path / "frame_000001.jpg"
    coarse_frame_path.touch()

    def mock_extract_frames_macro(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frame_000001.jpg").touch()
        return [
            {"frame_path": str(coarse_frame_path), "timestamp": 2.0, "frame_number": 1},
        ]

    captured_prompts: list[str] = []

    def mock_call_ollama(*args, **kwargs):
        prompt = args[3] if len(args) > 3 else kwargs.get("prompt", "")
        captured_prompts.append(prompt)
        return {
            "detected": True,
            "subject": "nudibranch",
            "confidence": 0.8,
            "quality_score": 8,
            "description": "Nudibranch seen",
            "timestamp": 2.0,
        }

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract_frames_macro),
        patch.object(scan, "call_ollama", side_effect=mock_call_ollama),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work")

    # Should have used macro template since tags match
    assert len(captured_prompts) >= 1
    prompt_used = captured_prompts[0]
    assert MACRO_TEMPLATE.coarse_prompt in prompt_used, (
        f"Expected macro template prompt, got: {prompt_used[:80]}..."
    )
    assert len(result) > 0


def test_process_clip_updates_state_activity(tmp_path: Path) -> None:
    """process_clip should heartbeat progress into the shared state file."""
    from utils.checkpoint import PipelineState

    video = tmp_path / "video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    coarse_frames = [
        {"frame_path": str(tmp_path / "frame_000001.jpg"), "timestamp": 2.0, "frame_number": 1},
    ]
    fine_frames = [
        {
            "frame_path": str(tmp_path / "fine_frame_000001.jpg"),
            "timestamp": 3.0,
            "frame_number": 1,
        },
    ]

    def mock_extract_frames(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return fine_frames if start_seconds is not None else coarse_frames

    mock_ollama_result = {
        "detected": True,
        "subject": "turtle",
        "confidence": 0.8,
        "quality_score": 8,
        "description": "Sea turtle",
    }
    response = MagicMock()
    response.json.return_value = {"response": json.dumps(mock_ollama_result)}
    client = ollama_client.OllamaClient("http://localhost:11434", "qwen2.5-vl:7b", 90)

    state = PipelineState(tmp_path / ".pipeline_state.json")
    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract_frames),
        patch.object(ollama_client.requests, "post", return_value=response),
    ):
        scan.process_clip(clip, config, tmp_path / "work", client, state=state)

    # Activity should describe the clip rather than a generic placeholder.
    assert "video.mp4" in (state.get("activity") or "")
    reloaded = PipelineState(tmp_path / ".pipeline_state.json")
    assert "video.mp4" in (reloaded.get("activity") or "")
    assert reloaded.get("activity_at")


def test_process_clip_honours_ollama_host_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OLLAMA_HOST should override the config host when building the client."""
    monkeypatch.setenv("OLLAMA_HOST", "plotter.local:11434")
    video = tmp_path / "video.mp4"
    video.touch()
    (tmp_path / "work").mkdir()
    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "clip_id": "c1",
        "fps": "25.0",
        "tags": [],
    }
    config = {
        "scan_parameters": {"coarse_fps": 0.5, "fine_fps": 2.0},
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    with (
        patch("utils.vlm_client.OllamaClient") as client_cls,
        patch.object(scan, "extract_frames", return_value=[]),
    ):
        scan.process_clip(clip, config, tmp_path / "work")

    assert client_cls.call_args.args[0] == "http://plotter.local:11434"


def test_main_output_dir_override_beats_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--output-dir must take effect instead of general.output_dir."""
    config_dir = tmp_path / "config_out"
    config_dir.mkdir()
    (config_dir / "manifest_raw.json").write_text(
        json.dumps({"project_name": "t", "total_clips": 0, "clips": []})
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'general:\n  output_dir: "{config_dir}"\n')

    override_dir = tmp_path / "override_out"
    override_dir.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        ["02_scan_footage.py", "--config", str(config_path), "--output-dir", str(override_dir)],
    )

    # The config folder holds a manifest, the override folder does not, so an
    # ignored override would scan successfully and never exit with code 1.
    with pytest.raises(SystemExit) as exc_info:
        scan.main()

    assert exc_info.value.code == 1


def test_process_clip_updates_progress_tasks(tmp_path: Path) -> None:
    """process_clip should create and advance progress tasks for coarse and fine passes."""
    video = tmp_path / "video.mp4"
    video.touch()
    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.5,
            "fine_padding_seconds": 2,
            "quality_threshold": 5,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }
    coarse_frames = [
        {"frame_path": str(tmp_path / "c1.jpg"), "timestamp": 1.0, "frame_number": 1},
    ]
    fine_frames = [
        {"frame_path": str(tmp_path / "f1.jpg"), "timestamp": 1.0, "frame_number": 1},
    ]
    (tmp_path / "c1.jpg").touch()
    (tmp_path / "f1.jpg").touch()

    def mock_extract(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        return fine_frames if start_seconds is not None else coarse_frames

    mock_ollama_result = {
        "detected": True,
        "subject": "shark",
        "confidence": 0.9,
        "quality_score": 8,
        "description": "Reef shark",
    }
    mock_progress = MagicMock()
    mock_progress.add_task.return_value = 1

    response = MagicMock()
    response.json.return_value = {"response": json.dumps(mock_ollama_result)}
    client = ollama_client.OllamaClient("http://localhost:11434", "qwen2.5-vl:7b", 90)

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(ollama_client.requests, "post", return_value=response),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work", client, progress=mock_progress)

    assert len(result) == 1
    assert mock_progress.add_task.call_count >= 2  # coarse task + fine window task
    assert mock_progress.advance.call_count >= 2


def test_create_scan_progress_layout() -> None:
    """create_scan_progress should use the shared console and bounded column layout."""
    from utils.logging_setup import get_console

    progress = scan.create_scan_progress()
    assert progress.console is get_console()
    assert len(progress.columns) > 0
    # Check that BarColumn has a bounded width to prevent terminal line wrapping
    bar_cols = [c for c in progress.columns if isinstance(c, scan.BarColumn)]
    assert len(bar_cols) == 1
    assert bar_cols[0].bar_width is not None


def test_process_clip_passes_scale_parameters(tmp_path: Path) -> None:
    """process_clip should pass coarse_scale and fine_scale to extract_frames."""
    video = tmp_path / "video.mp4"
    video.touch()
    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "coarse_scale": "640:-2",
            "fine_fps": 2.0,
            "fine_scale": "1280:-2",
            "coarse_confidence_threshold": 0.5,
            "fine_padding_seconds": 2,
            "quality_threshold": 5,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }
    extracted_scales = []

    def mock_extract(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        extracted_scales.append(scale)
        frame = tmp_path / f"f_{len(extracted_scales)}.jpg"
        frame.touch()
        return [{"frame_path": str(frame), "timestamp": 1.0, "frame_number": 1}]

    mock_ollama_result = {
        "detected": True,
        "subject": "turtle",
        "confidence": 0.9,
        "quality_score": 8,
        "description": "Sea turtle",
    }
    response = MagicMock()
    response.json.return_value = {"response": json.dumps(mock_ollama_result)}
    client = ollama_client.OllamaClient("http://localhost:11434", "qwen2.5-vl:7b", 90)

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(ollama_client.requests, "post", return_value=response),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work", client)

    assert len(result) == 1
    # First call (coarse) used coarse_scale, second call (fine) used fine_scale
    assert extracted_scales[0] == "640:-2"
    assert extracted_scales[1] == "1280:-2"


def test_save_detections_atomic(tmp_path: Path) -> None:
    """save_detections_atomic should write detections cleanly and atomically."""
    from utils.models import DetectionsRefined, RefinedDetection

    det = RefinedDetection(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        start_time=0.0,
        end_time=5.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="test",
    )
    refined = DetectionsRefined(total_clips=1, total_detections=1, detections=[det])
    out_file = tmp_path / "detections_refined.json"

    scan.save_detections_atomic(out_file, refined)

    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["total_clips"] == 1
    assert loaded["total_detections"] == 1
    assert loaded["detections"][0]["subject"] == "turtle"


def test_scan_main_incremental_and_resume(tmp_path: Path) -> None:
    """main should save detections after each clip and skip done clips on resume."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_raw.json"
    manifest_data = {
        "clips": [
            {"file_name": "clip1.mp4", "file_path": str(tmp_path / "clip1.mp4"), "fps": 25.0},
            {"file_name": "clip2.mp4", "file_path": str(tmp_path / "clip2.mp4"), "fps": 25.0},
        ]
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    (tmp_path / "clip1.mp4").touch()
    (tmp_path / "clip2.mp4").touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text("general:\n  output_dir: " + str(output_dir) + "\n", encoding="utf-8")

    state_path = output_dir / ".pipeline_state.json"

    from utils.models import RefinedDetection

    det1 = RefinedDetection(
        clip_id="c1",
        file_name="clip1.mp4",
        file_path=str(tmp_path / "clip1.mp4"),
        start_time=0.0,
        end_time=2.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="turtle",
    )
    det2 = RefinedDetection(
        clip_id="c2",
        file_name="clip2.mp4",
        file_path=str(tmp_path / "clip2.mp4"),
        start_time=0.0,
        end_time=3.0,
        subject="shark",
        confidence=0.95,
        quality_score=9,
        description="shark",
    )

    saved_states = []

    def mock_save_atomic(path, refined):
        saved_states.append(list(refined.detections))
        # actually write it
        path.write_text(json.dumps(refined.model_dump()), encoding="utf-8")

    call_count = 0

    def mock_process_clip(clip, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return [det1] if clip["file_name"] == "clip1.mp4" else [det2]

    # Run pass 1: both clips processed
    with (
        patch.object(scan, "process_clip", side_effect=mock_process_clip),
        patch.object(scan, "save_detections_atomic", side_effect=mock_save_atomic),
        patch(
            "sys.argv",
            [
                "02_scan_footage.py",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--state-path",
                str(state_path),
            ],
        ),
    ):
        scan.main()

    # Detections saved after each clip
    assert len(saved_states) >= 2
    assert len(saved_states[0]) == 1  # after clip1
    assert len(saved_states[1]) == 2  # after clip2
    assert call_count == 2

    # Run pass 2 with --resume: clip1 and clip2 should be skipped
    call_count = 0
    with (
        patch.object(scan, "process_clip", side_effect=mock_process_clip),
        patch(
            "sys.argv",
            [
                "02_scan_footage.py",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--state-path",
                str(state_path),
                "--resume",
            ],
        ),
    ):
        scan.main()

    # No new clips processed because both were completed
    assert call_count == 0
    # Final file still contains both detections
    final_data = json.loads((output_dir / "detections_refined.json").read_text(encoding="utf-8"))
    assert len(final_data["detections"]) == 2


def test_scan_main_keyboard_interrupt_handles_gracefully(tmp_path: Path) -> None:
    """main should catch KeyboardInterrupt, save existing detections, clean up work_dir, and exit with code 130."""
    from utils.models import RefinedDetection

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_raw.json"
    manifest_data = {
        "clips": [
            {"file_name": "clip1.mp4", "file_path": str(tmp_path / "clip1.mp4"), "fps": 25.0},
            {"file_name": "clip2.mp4", "file_path": str(tmp_path / "clip2.mp4"), "fps": 25.0},
        ]
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    (tmp_path / "clip1.mp4").touch()
    (tmp_path / "clip2.mp4").touch()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"general:\n  output_dir: {output_dir}\n", encoding="utf-8")

    det1 = RefinedDetection(
        clip_id="c1",
        file_name="clip1.mp4",
        file_path=str(tmp_path / "clip1.mp4"),
        start_time=0.0,
        end_time=2.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="turtle",
    )

    def mock_process_clip(clip, *args, **kwargs):
        if "pass0_collector" in kwargs and kwargs["pass0_collector"] is not None:
            from utils.models import ClipClassification

            kwargs["pass0_collector"].append(
                ClipClassification(
                    clip_id="c1",
                    file_name="clip1.mp4",
                    file_path=str(tmp_path / "clip1.mp4"),
                    domain="underwater",
                )
            )
        if clip["file_name"] == "clip1.mp4":
            return [det1]
        raise KeyboardInterrupt

    cleaned_dirs = []

    def mock_cleanup(d):
        cleaned_dirs.append(Path(d))

    with (
        patch.object(scan, "process_clip", side_effect=mock_process_clip),
        patch.object(scan, "cleanup_work_dir", side_effect=mock_cleanup),
        patch(
            "sys.argv",
            [
                "02_scan_footage.py",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ],
        ),
    ):
        with pytest.raises(SystemExit) as exc_info:
            scan.main()
        assert exc_info.value.code == 130

    # Ensure detections from clip1 were saved before exit
    saved_data = json.loads((output_dir / "detections_refined.json").read_text(encoding="utf-8"))
    assert len(saved_data["detections"]) == 1
    assert saved_data["detections"][0]["subject"] == "turtle"
    # Ensure classifications were saved before exit
    assert (output_dir / "classifications.json").exists()
    # Ensure cleanup_work_dir was called
    assert len(cleaned_dirs) > 0


def test_coarse_scan_rejects_upside_down_or_dangling_by_quality_threshold(tmp_path: Path) -> None:
    """Pass 1 coarse scan should reject detected frames when quality_score is below coarse_quality_threshold."""
    video = tmp_path / "inverted_video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "inverted_video.mp4",
        "tags": [],
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "coarse_quality_threshold": 4,
            "fine_padding_seconds": 4,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    coarse_frame_path = tmp_path / "frame_000001.jpg"
    coarse_frame_path.touch()

    def mock_extract(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        return [
            {"frame_path": str(coarse_frame_path), "timestamp": 2.0, "frame_number": 1},
        ]

    # Model returned detected=True but low quality (e.g. 2 because camera is dangling / upside down)
    def mock_call(*args, **kwargs):
        return {
            "detected": True,
            "subject": "coral",
            "confidence": 0.9,
            "quality_score": 2,
            "description": "Camera dangling upside down, coral barely visible",
            "timestamp": 2.0,
        }

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work")

    # Frame rejected in coarse scan -> no candidate windows -> no fine detections
    assert result == []


def test_coarse_scan_accepts_good_orientation_frame(tmp_path: Path) -> None:
    """Pass 1 coarse scan should accept detected frames when quality_score meets coarse_quality_threshold."""
    video = tmp_path / "good_video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "good_video.mp4",
        "tags": [],
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "coarse_quality_threshold": 4,
            "fine_padding_seconds": 2,
            "quality_threshold": 7,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
            "timeout_seconds": 90,
        },
    }

    frame_path = tmp_path / "frame_000001.jpg"
    frame_path.touch()

    def mock_extract(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            {"frame_path": str(frame_path), "timestamp": 2.0, "frame_number": 1},
        ]

    def mock_call(*args, **kwargs):
        return {
            "detected": True,
            "subject": "turtle",
            "confidence": 0.85,
            "quality_score": 8,
            "description": "Upright clear shot of sea turtle",
            "timestamp": 2.0,
        }

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        result = scan.process_clip(clip, config, tmp_path / "work")

    assert len(result) > 0
    assert result[0].subject == "turtle"


def test_scan_main_archives_to_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--run-dir should archive detections and manifest into specified folder."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_raw.json"
    manifest_path.write_text(json.dumps({"clips": []}), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"general:\n  output_dir: {output_dir}\n", encoding="utf-8")

    run_dir = tmp_path / "custom_run"

    monkeypatch.setattr(
        "sys.argv",
        [
            "02_scan_footage.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-dir",
            str(run_dir),
        ],
    )

    scan.main()

    assert (output_dir / "detections_refined.json").exists()
    assert (run_dir / "detections_refined.json").exists()
    assert (run_dir / "manifest_raw.json").exists()


def test_scan_main_archives_with_archive_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--archive flag should create timestamped directory under <output>/runs/run-<stamp>/."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_raw.json"
    manifest_path.write_text(json.dumps({"clips": []}), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"general:\n  output_dir: {output_dir}\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "02_scan_footage.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--archive",
        ],
    )

    scan.main()

    runs_dir = output_dir / "runs"
    assert runs_dir.exists()
    run_dirs = list(runs_dir.iterdir())
    assert len(run_dirs) == 1
    assert run_dirs[0].name.startswith("run-")
    assert (run_dirs[0] / "detections_refined.json").exists()
    assert (run_dirs[0] / "manifest_raw.json").exists()


def test_scan_main_passes_use_ramdisk_to_get_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main should pass use_ramdisk from scan_parameters or processing to get_work_dir."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest_raw.json"
    manifest_path.write_text(json.dumps({"clips": []}), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"general:\n  output_dir: {output_dir}\nscan_parameters:\n  use_ramdisk: false\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["02_scan_footage.py", "--config", str(config_path), "--output-dir", str(output_dir)],
    )

    with (
        patch.object(scan, "get_work_dir", return_value=tmp_path / "work") as mock_get_work,
        patch.object(scan, "cleanup_work_dir"),
    ):
        scan.main()
        mock_get_work.assert_called_once_with(use_ramdisk=False)


def test_scan_main_emits_run_stats_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main should emit run_stats.json and update state with scan_stats."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "project_name": "Test",
            "total_clips": 1,
            "clips": [
                {
                    "clip_id": "clip1",
                    "file_name": "video1.mp4",
                    "file_path": str(tmp_path / "video1.mp4"),
                    "duration": 42.0,
                }
            ],
        }),
        encoding="utf-8",
    )

    state_path = output_dir / "pipeline_state.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"general:\n  output_dir: {output_dir}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "02_scan_footage.py",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--state-path",
            str(state_path),
        ],
    )

    with (
        patch.object(scan, "get_work_dir", return_value=tmp_path / "work"),
        patch.object(scan, "cleanup_work_dir"),
        patch.object(scan, "process_clip", return_value=[]),
    ):
        scan.main()

    stats_file = output_dir / "run_stats.json"
    assert stats_file.exists()
    data = json.loads(stats_file.read_text(encoding="utf-8"))
    assert data["video_count"] == 1
    assert data["total_video_duration"] == 42.0
    assert "phase1" in data
    assert "phase2" in data

    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    assert "scan_stats" in state_data
    assert state_data["scan_stats"]["video_count"] == 1


def test_process_clip_with_pass0_classification(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """process_clip with domain='auto' should classify clip in Pass 0, extracting frames under [Pass 0] before [Pass 1] scan."""
    caplog.set_level(logging.INFO)
    from utils.stats import Phase0Stats

    video = tmp_path / "moto.mp4"
    video.touch()
    clip = {
        "clip_id": "moto_1",
        "file_name": "moto.mp4",
        "file_path": str(video),
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "domain": "auto",
            "classify_clips": "auto",
            "classify_frames_count": 2,
            "coarse_fps": 0.5,
            "fine_fps": 1.0,
            "coarse_confidence_threshold": 0.7,
            "coarse_quality_threshold": 4,
            "quality_threshold": 6,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }

    coarse_frame = tmp_path / "c_frame.jpg"
    coarse_frame.touch()
    fine_frame = tmp_path / "f_frame.jpg"
    fine_frame.touch()

    def mock_extract(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if start_seconds is not None:
            return [{"frame_path": str(fine_frame), "timestamp": 2.0, "frame_number": 1}]
        return [
            {"frame_path": str(coarse_frame), "timestamp": 2.0, "frame_number": 1},
            {"frame_path": str(coarse_frame), "timestamp": 4.0, "frame_number": 2},
        ]

    calls_by_phase: dict[str, list[dict]] = {"classify": [], "coarse": [], "fine": []}

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        if phase == "classify":
            calls_by_phase["classify"].append({"prompt": prompt})
            return {
                "domain": "motorcycle",
                "confidence": 0.95,
                "environment": "mountain road",
                "primary_subjects": ["motorcycle", "twisties"],
            }
        if phase == "coarse":
            calls_by_phase["coarse"].append({"prompt": prompt})
            return {
                "detected": True,
                "confidence": 0.88,
                "quality_score": 8,
                "subject": "hairpin corner lean",
                "category": "moto_riding",
            }
        if phase == "fine":
            calls_by_phase["fine"].append({"prompt": prompt})
            return {
                "detected": True,
                "confidence": 0.92,
                "quality_score": 9,
                "subject": "deep lean into hairpin",
                "description": "Smooth aggressive lean through curve",
            }
        return None

    p0_stats = Phase0Stats()
    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        detections = scan.process_clip(
            clip,
            config,
            tmp_path / "work",
            phase0_stats=p0_stats,
        )

    assert len(detections) == 1
    det = detections[0]
    assert det.domain == "motorcycle"
    assert det.template == "moto_riding"
    assert det.color == "Orange"
    assert "hairpin" in det.subject

    assert len(calls_by_phase["classify"]) >= 1
    assert len(calls_by_phase["coarse"]) >= 1
    assert len(calls_by_phase["fine"]) >= 1

    assert p0_stats.clips_classified == 1
    assert p0_stats.domain_counts.get("motorcycle") == 1

    # Verify frame extraction logging is under [Pass 0] and Pass 1 follows domain resolution
    assert "[Pass 0] Extracting frames @ 0.5 FPS: moto.mp4" in caplog.text
    assert "[Pass 0] Extracted 2 frames from moto.mp4" in caplog.text
    assert "[Pass 0] Classified moto.mp4 as domain 'motorcycle'" in caplog.text
    assert "[Pass 1] Coarse scan @ 0.5 FPS (motorcycle): moto.mp4" in caplog.text


def test_process_clip_domain_override(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """When scan_parameters.domain is explicitly set, Pass 0 is bypassed and domain coarse prompt is used."""
    caplog.set_level(logging.INFO)
    video = tmp_path / "moto.mp4"
    video.touch()
    clip = {
        "clip_id": "moto_override",
        "file_name": "moto.mp4",
        "file_path": str(video),
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "domain": "motorcycle",
            "coarse_fps": 0.5,
            "fine_fps": 1.0,
            "coarse_confidence_threshold": 0.7,
            "coarse_quality_threshold": 4,
            "quality_threshold": 6,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }

    frame = tmp_path / "f.jpg"
    frame.touch()

    calls: list[str] = []

    def mock_extract(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return [{"frame_path": str(frame), "timestamp": 2.0, "frame_number": 1}]

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        calls.append(phase)
        return {
            "detected": True,
            "confidence": 0.85,
            "quality_score": 8,
            "subject": "riding",
            "category": "moto_riding",
        }

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        detections = scan.process_clip(clip, config, tmp_path / "work")

    # Pass 0 ('classify') was bypassed because domain='motorcycle' was forced
    assert "classify" not in calls
    assert "coarse" in calls
    assert len(detections) >= 1
    assert detections[0].domain == "motorcycle"

    # Verify frame extraction was logged under [Pass 1] since domain was already resolved
    assert "[Pass 0] Using configured domain 'motorcycle' for moto.mp4" in caplog.text
    assert "[Pass 1] Coarse scan @ 0.5 FPS (motorcycle): moto.mp4" in caplog.text
    assert "[Pass 1] Extracted 1 frames from moto.mp4" in caplog.text
    assert "[Pass 0] Extracting frames" not in caplog.text


def test_process_clip_subject_routing_multicategory(tmp_path: Path) -> None:
    """Candidate windows with distinct subjects in the same clip should route to matching fine templates."""
    video = tmp_path / "moto_multi.mp4"
    video.touch()
    clip = {
        "clip_id": "moto_multi",
        "file_name": "moto_multi.mp4",
        "file_path": str(video),
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "domain": "motorcycle",
            "coarse_fps": 0.5,
            "fine_fps": 1.0,
            "coarse_confidence_threshold": 0.7,
            "coarse_quality_threshold": 4,
            "fine_padding_seconds": 1,
            "quality_threshold": 6,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }

    c1 = tmp_path / "c1.jpg"
    c2 = tmp_path / "c2.jpg"
    c1.touch()
    c2.touch()

    # Two distinct windows far apart: t=2.0 (riding) and t=20.0 (scenic)
    def mock_extract(
        video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if start_seconds is not None:
            # Fine pass: single frame per window
            return [{"frame_path": str(c1), "timestamp": start_seconds + 0.5, "frame_number": 1}]
        return [
            {"frame_path": str(c1), "timestamp": 2.0, "frame_number": 1},
            {"frame_path": str(c2), "timestamp": 20.0, "frame_number": 10},
        ]

    fine_prompts_seen: list[str] = []

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        if phase == "coarse":
            # If timestamp is around 2.0 -> cornering (moto_riding)
            # If timestamp is around 20.0 -> mountain pass vista (moto_scenic)
            if "frame_number" in str(img) or "c1" in str(img):
                return {
                    "detected": True,
                    "confidence": 0.9,
                    "quality_score": 8,
                    "subject": "cornering hairpin",
                    "category": "moto_riding",
                }
            return {
                "detected": True,
                "confidence": 0.9,
                "quality_score": 8,
                "subject": "mountain pass vista",
                "category": "moto_scenic",
            }
        if phase == "fine":
            fine_prompts_seen.append(prompt)
            return {
                "detected": True,
                "confidence": 0.9,
                "quality_score": 8,
                "subject": "highlight",
                "description": "fine highlight",
            }
        return None

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        detections = scan.process_clip(clip, config, tmp_path / "work")

    assert len(detections) == 2
    templates_used = {d.template for d in detections}
    colors_used = {d.color for d in detections}
    assert "moto_riding" in templates_used
    assert "moto_scenic" in templates_used
    assert "Orange" in colors_used
    assert "Blue" in colors_used


def test_process_clip_unusable_early_exit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """process_clip should immediately abort and skip coarse extraction if Pass 0 classifies footage as unusable."""
    caplog.set_level(logging.INFO)
    from utils.stats import Phase0Stats

    video = tmp_path / "in_bag.mp4"
    video.touch()
    clip = {
        "clip_id": "bag_1",
        "file_name": "in_bag.mp4",
        "file_path": str(video),
        "fps": 25.0,
        "duration": 600.0,
    }
    config = {
        "scan_parameters": {
            "domain": "auto",
            "classify_clips": "auto",
            "classify_frames_count": 3,
            "coarse_fps": 0.5,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }

    sample_frame = tmp_path / "sample.jpg"
    sample_frame.touch()

    coarse_extracted = False

    def mock_extract(video_path, output_dir, *args, **kwargs):
        nonlocal coarse_extracted
        if "coarse" in str(output_dir):
            coarse_extracted = True
        return [{"frame_path": str(sample_frame), "timestamp": 1.0, "frame_number": 1}]

    def mock_targeted(video_path, output_dir, timestamps, scale=None):
        return [
            {"frame_path": str(sample_frame), "timestamp": ts, "frame_number": idx}
            for idx, ts in enumerate(timestamps, start=1)
        ]

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        if phase == "classify":
            return {
                "domain": "unusable",
                "confidence": 0.99,
                "usable": False,
                "quality_score": 1,
                "reason": "inside gear bag",
            }
        return None

    p0_stats = Phase0Stats()
    with (
        patch.object(scan, "extract_targeted_frames", side_effect=mock_targeted),
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        detections = scan.process_clip(
            clip,
            config,
            tmp_path / "work",
            phase0_stats=p0_stats,
        )

    assert detections == []
    assert not coarse_extracted
    assert p0_stats.clips_unusable == 1
    assert "unusable camera footage" in caplog.text


def test_process_clip_non_target_domain_early_exit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """process_clip should immediately abort and skip coarse extraction if detected domain does not match configured domain."""
    caplog.set_level(logging.INFO)
    from utils.stats import Phase0Stats

    video = tmp_path / "motorcycle.mp4"
    video.touch()
    clip = {
        "clip_id": "moto_2",
        "file_name": "motorcycle.mp4",
        "file_path": str(video),
        "fps": 25.0,
        "duration": 300.0,
    }
    config = {
        "scan_parameters": {
            "domain": "underwater",
            "classify_clips": "always",
            "classify_frames_count": 3,
            "coarse_fps": 0.5,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }

    sample_frame = tmp_path / "sample.jpg"
    sample_frame.touch()

    coarse_extracted = False

    def mock_extract(video_path, output_dir, *args, **kwargs):
        nonlocal coarse_extracted
        if "coarse" in str(output_dir):
            coarse_extracted = True
        return [{"frame_path": str(sample_frame), "timestamp": 1.0, "frame_number": 1}]

    def mock_targeted(video_path, output_dir, timestamps, scale=None):
        return [
            {"frame_path": str(sample_frame), "timestamp": ts, "frame_number": idx}
            for idx, ts in enumerate(timestamps, start=1)
        ]

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        if phase == "classify":
            return {
                "domain": "motorcycle",
                "confidence": 0.95,
                "usable": True,
                "quality_score": 8,
            }
        return None

    p0_stats = Phase0Stats()
    with (
        patch.object(scan, "extract_targeted_frames", side_effect=mock_targeted),
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        detections = scan.process_clip(
            clip,
            config,
            tmp_path / "work",
            phase0_stats=p0_stats,
        )

    assert detections == []
    assert not coarse_extracted
    assert p0_stats.clips_skipped_domain == 1
    assert "non-target domain 'motorcycle' (configured: 'underwater')" in caplog.text


def test_process_clip_reuses_prefetched_coarse_frames_and_triggers_callback(
    tmp_path: Path,
) -> None:
    """process_clip should reuse prefetched_coarse_frames and invoke on_coarse_start callback."""
    video = tmp_path / "dive.mp4"
    video.touch()
    clip = {
        "clip_id": "dive_1",
        "file_name": "dive.mp4",
        "file_path": str(video),
        "fps": 25.0,
        "domain": "underwater",
    }
    config = {
        "scan_parameters": {
            "domain": "underwater",
            "classify_clips": "off",
            "coarse_fps": 0.5,
            "fine_fps": 1.0,
            "coarse_confidence_threshold": 0.7,
            "coarse_quality_threshold": 4,
            "quality_threshold": 6,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }

    c_frame = tmp_path / "c_frame.jpg"
    c_frame.touch()
    f_frame = tmp_path / "f_frame.jpg"
    f_frame.touch()

    coarse_extracted = False

    def mock_extract(video_path, output_dir, *args, **kwargs):
        nonlocal coarse_extracted
        if "coarse" in str(output_dir):
            coarse_extracted = True
        return [{"frame_path": str(f_frame), "timestamp": 2.0, "frame_number": 1}]

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        if phase == "coarse":
            return {
                "detected": True,
                "confidence": 0.85,
                "quality_score": 7,
                "subject": "sea turtle",
                "category": "wildlife",
            }
        if phase == "fine":
            return {
                "detected": True,
                "confidence": 0.9,
                "quality_score": 8,
                "subject": "green sea turtle",
                "description": "turtle swimming",
            }
        return None

    callback_called = False

    def on_coarse_start():
        nonlocal callback_called
        callback_called = True

    prefetched = [
        {"frame_path": str(c_frame), "timestamp": 2.0, "frame_number": 1},
        {"frame_path": str(c_frame), "timestamp": 4.0, "frame_number": 2},
    ]

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        detections = scan.process_clip(
            clip,
            config,
            tmp_path / "work",
            prefetched_coarse_frames=prefetched,
            on_coarse_start=on_coarse_start,
        )

    assert len(detections) == 1
    assert not coarse_extracted
    assert callback_called


def test_extract_classification_frames_duration_sampling(tmp_path: Path) -> None:
    """extract_classification_frames should calculate dynamic sample count and timestamps from video duration."""
    video = tmp_path / "long_video.mp4"
    video.touch()

    captured_timestamps: list[float] = []

    def mock_targeted(video_path, output_dir, timestamps, scale=None):
        nonlocal captured_timestamps
        captured_timestamps = list(timestamps)
        return [
            {"frame_path": str(tmp_path / "f.jpg"), "timestamp": t, "frame_number": i}
            for i, t in enumerate(timestamps, 1)
        ]

    with patch.object(scan, "extract_targeted_frames", side_effect=mock_targeted):
        frames = scan.extract_classification_frames(
            file_path=video,
            output_dir=tmp_path / "pass0",
            count=3,
            duration=1200.0,  # 20 min -> 4 frames (1200 / 300 = 4)
            interval_seconds=300.0,
            max_count=8,
        )

    assert len(frames) == 4
    assert len(captured_timestamps) == 4
    # Evenly spaced across 1200s (step = 1200 / 5 = 240)
    assert captured_timestamps == [240.0, 480.0, 720.0, 960.0]


def test_process_clip_traces_and_natural_window_sizing(tmp_path: Path) -> None:
    """process_clip should populate pass1 and pass2 collectors and apply natural window sizing."""
    from utils.models import Pass1ClipTrace, Pass2ClipTrace

    video = tmp_path / "test_traces.mp4"
    video.touch()
    clip = {
        "clip_id": "trace_clip_1",
        "file_name": "test_traces.mp4",
        "file_path": str(video),
        "fps": 25.0,
        "domain": "underwater",
    }
    config = {
        "scan_parameters": {
            "domain": "underwater",
            "classify_clips": "off",
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.65,
            "coarse_quality_threshold": 4,
            "quality_threshold": 7,
            "fine_pre_roll_seconds": 1.5,
            "fine_post_roll_seconds": 2.0,
            "merge_gap_seconds": 1.5,
            "max_clip_duration": 12.0,
        },
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5-vl:7b",
        },
    }

    c_frame = tmp_path / "c.jpg"
    c_frame.touch()
    f_frame = tmp_path / "f.jpg"
    f_frame.touch()

    def mock_extract(video_path, output_dir, *args, **kwargs):
        if "fine" in str(output_dir):
            return [{"frame_path": str(f_frame), "timestamp": 5.0, "frame_number": 1}]
        return [{"frame_path": str(c_frame), "timestamp": 4.0, "frame_number": 1}]

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        if phase == "coarse":
            return {
                "detected": True,
                "confidence": 0.9,
                "quality_score": 8,
                "subject": "green sea turtle",
                "category": "pelagics",
                "story_role": "hero_subject",
            }
        return {
            "detected": True,
            "confidence": 0.95,
            "quality_score": 9,
            "subject": "green sea turtle",
            "category": "pelagics",
            "story_role": "hero_subject",
            "description": "Green sea turtle gliding over reef",
        }

    pass1_collector: list[Pass1ClipTrace] = []
    pass2_collector: list[Pass2ClipTrace] = []

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        detections = scan.process_clip(
            clip,
            config,
            tmp_path / "work",
            pass1_collector=pass1_collector,
            pass2_collector=pass2_collector,
        )

    assert len(detections) == 1
    det = detections[0]
    # Frame timestamp was 5.0: start_sec = 5.0 - 1.5 = 3.5, end_sec = 5.0 + 2.0 = 7.0
    assert det.start_time == 3.5
    assert det.end_time == 7.0
    assert det.story_role == "hero_subject"
    assert det.detection_id == f"trace_clip_1_{3.5:08.2f}_{7.0:08.2f}"
    assert det.sort_key == f"test_traces.mp4:{3.5:08.2f}:{7.0:08.2f}:trace_clip_1"

    # Verify Pass 1 trace
    assert len(pass1_collector) == 1
    p1 = pass1_collector[0]
    assert p1.clip_id == "trace_clip_1"
    assert p1.source_file == "test_traces.mp4"
    assert len(p1.frame_traces) == 1
    assert p1.frame_traces[0].detection_id == f"trace_clip_1_{4.0:08.2f}_{4.0:08.2f}"
    assert p1.frame_traces[0].passed_gate is True

    # Verify Pass 2 trace
    assert len(pass2_collector) == 1
    p2 = pass2_collector[0]
    assert p2.clip_id == "trace_clip_1"
    assert p2.fine_pre_roll_seconds == 1.5
    assert p2.fine_post_roll_seconds == 2.0
    assert len(p2.frame_traces) == 1
    assert p2.frame_traces[0].detection_id == f"trace_clip_1_{3.5:08.2f}_{7.0:08.2f}"
    assert p2.frame_traces[0].passed_gate is True


def test_process_clip_on_trace_flush_called(tmp_path: Path) -> None:
    """process_clip should invoke on_trace_flush callback across scan phases."""
    video = tmp_path / "test_flush.mp4"
    video.touch()
    clip = {
        "file_path": str(video),
        "file_name": "test_flush.mp4",
        "clip_id": "flush_clip_1",
        "fps": 25.0,
        "duration": "00:00:10.00",
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "fine_fps": 2.0,
            "coarse_confidence_threshold": 0.5,
            "quality_threshold": 5,
        },
        "domain": "diving",
    }
    coarse_frames = [
        {"frame_path": str(tmp_path / "c1.jpg"), "timestamp": 2.0, "frame_number": 50},
    ]
    (tmp_path / "c1.jpg").touch()

    def mock_extract(video_path, output_dir, *args, **kwargs):
        return coarse_frames

    def mock_call(*args, **kwargs):
        return {
            "detected": False,
            "confidence": 0.1,
            "quality_score": 2,
            "subject": "",
        }

    flush_calls: list[str] = []
    from utils.models import ClipClassification, Pass1ClipTrace

    pass0_collector: list[ClipClassification] = []
    pass1_collector: list[Pass1ClipTrace] = []

    with (
        patch.object(scan, "extract_frames", side_effect=mock_extract),
        patch.object(scan, "call_ollama", side_effect=mock_call),
    ):
        scan.process_clip(
            clip,
            config,
            tmp_path / "work",
            pass0_collector=pass0_collector,
            pass1_collector=pass1_collector,
            on_trace_flush=flush_calls.append,
        )

    assert "pass0" in flush_calls
    assert "pass1" in flush_calls


def _make_prescreen_record(**overrides):
    """Build a minimal PrescreenRecord for trace-projection tests."""
    from utils.models import PrescreenInterval, PrescreenRecord

    payload = {
        "clip_id": "ps_clip",
        "source_file": "clip.mp4",
        "interval": PrescreenInterval(start=10.0, end=20.0),
        "action": "reject",
        "telemetry_score": 0.42,
        "hsv_entropy": 0.75,
        "laplacian_sharpness": 123.4,
        "audio_peak_db": -8.5,
        "audio_spike_triggered": True,
    }
    payload.update(overrides)
    return PrescreenRecord(**payload)


def test_prescreen_trace_fields_empty_when_screening_disabled() -> None:
    """A disabled funnel yields no trace fields at all."""
    assert scan._prescreen_trace_fields(None, 12.0) == {}


def test_prescreen_trace_fields_action_only_outside_records() -> None:
    """Timestamps outside every window report the default keep action only."""
    from utils.prescreen import PrescreenResult

    result = PrescreenResult(records=[_make_prescreen_record()])

    assert scan._prescreen_trace_fields(result, 99.0) == {"prescreen_action": "keep"}


def test_prescreen_trace_fields_projects_covering_record() -> None:
    """A timestamp inside a window carries that record's metrics."""
    from utils.prescreen import PrescreenResult

    result = PrescreenResult(records=[_make_prescreen_record()])
    fields = scan._prescreen_trace_fields(result, 15.0)

    assert fields == {
        "prescreen_action": "reject",
        "telemetry_score": 0.42,
        "hsv_entropy": 0.75,
        "laplacian_sharpness": 123.4,
        "audio_peak_db": -8.5,
        "audio_spike_triggered": True,
    }


def test_prescreen_trace_fields_interval_bounds_are_inclusive() -> None:
    """Boundary timestamps resolve to the window covering them."""
    from utils.prescreen import PrescreenResult

    result = PrescreenResult(records=[_make_prescreen_record()])

    assert scan._prescreen_trace_fields(result, 10.0)["prescreen_action"] == "reject"
    assert scan._prescreen_trace_fields(result, 20.0)["prescreen_action"] == "reject"
    assert scan._prescreen_trace_fields(result, 20.001)["prescreen_action"] == "keep"


def test_prescreen_clip_returns_none_when_disabled(tmp_path: Path) -> None:
    """A disabled config skips FFmpeg, telemetry, and screening entirely."""
    from utils.prescreen import PrescreenConfig

    with (
        patch.object(scan, "load_telemetry") as mock_telemetry,
        patch.object(scan, "screen_clip") as mock_screen,
    ):
        result, track = scan._prescreen_clip(
            {"clip_id": "c1", "file_name": "clip.mp4", "duration": "12.0"},
            str(tmp_path / "clip.mp4"),
            PrescreenConfig(enabled=False),
        )

    assert result is None
    assert track is None
    mock_telemetry.assert_not_called()
    mock_screen.assert_not_called()


def test_prescreen_clip_loads_telemetry_and_screens(tmp_path: Path) -> None:
    """Enabled screening loads the telemetry track and forwards clip metadata."""
    from utils.prescreen import PrescreenConfig, PrescreenResult

    track = MagicMock(name="track")
    expected = PrescreenResult(duration=12.0)
    cache_dir = tmp_path / "telemetry"

    with (
        patch.object(scan, "load_telemetry", return_value=track) as mock_telemetry,
        patch.object(scan, "screen_clip", return_value=expected) as mock_screen,
    ):
        result, got_track = scan._prescreen_clip(
            {
                "clip_id": "c1",
                "file_name": "clip.mp4",
                "duration": "12.5",
                "telemetry_type": "gpmf",
                "hilight_tags": [3.0],
                "bitrate_density": 1.5,
                "is_high_fps": True,
            },
            str(tmp_path / "clip.mp4"),
            PrescreenConfig(enabled=True),
            telemetry_cache_dir=cache_dir,
        )

    assert result is expected
    assert got_track is track
    assert mock_telemetry.call_args.kwargs["telemetry_type"] == "gpmf"
    assert mock_telemetry.call_args.kwargs["cache_path"] == cache_dir / "c1.telemetry.bin"
    kwargs = mock_screen.call_args.kwargs
    assert kwargs["clip_id"] == "c1"
    assert kwargs["hilight_tags"] == [3.0]
    assert kwargs["bitrate_density"] == 1.5
    assert kwargs["is_high_fps"] is True
    assert kwargs["track"] is track


def test_analyze_contact_sheets_batches_and_maps_timestamps(tmp_path: Path) -> None:
    """Frames are grouped into full sheets and results keyed by tile timestamp."""
    frames = [
        {"frame_path": str(tmp_path / f"f{index}.jpg"), "timestamp": float(index)}
        for index in range(4)
    ]
    tiles = [{"detected": True, "subject": f"t{index}"} for index in range(4)]

    with (
        patch.object(scan, "compose_frame_mosaic", return_value=str(tmp_path / "s0.jpg")) as mosaic,
        patch.object(scan, "call_ollama", return_value={"tiles": tiles}),
        patch.object(scan, "normalize_vlm_sheet_response", return_value=tiles),
    ):
        results, meta = scan._analyze_contact_sheets(
            frames,
            sheet_dir=tmp_path / "sheets",
            prompt="P",
            system_prompt="S",
            ollama_host="http://x",
            ollama_model="m",
            timeout=5.0,
            client=None,
        )

    assert mosaic.call_count == 1
    assert mosaic.call_args.args[0] == [frame["frame_path"] for frame in frames]
    assert set(results) == {0.0, 1.0, 2.0, 3.0}
    assert results[2.0]["subject"] == "t2"
    assert meta[3.0] == {
        "contact_sheet_path": str(tmp_path / "s0.jpg"),
        "tile_index": 3,
        "tile_timestamp": 3.0,
        "tile_count": 4,
    }


def test_analyze_contact_sheets_skips_partial_batch_and_failures(tmp_path: Path) -> None:
    """Partial batches and unusable sheets are left for the per-frame fallback."""
    frames = [
        {"frame_path": str(tmp_path / f"f{index}.jpg"), "timestamp": float(index)}
        for index in range(5)
    ]

    with (
        patch.object(scan, "compose_frame_mosaic", return_value=None),
        patch.object(scan, "call_ollama") as mock_call,
    ):
        assert scan._analyze_contact_sheets(
            frames,
            sheet_dir=tmp_path / "sheets",
            prompt="P",
            system_prompt="S",
            ollama_host="http://x",
            ollama_model="m",
            timeout=5.0,
            client=None,
        ) == ({}, {})
    mock_call.assert_not_called()

    with (
        patch.object(scan, "compose_frame_mosaic", return_value=str(tmp_path / "s0.jpg")),
        patch.object(scan, "call_ollama", return_value={"tiles": []}),
        patch.object(scan, "normalize_vlm_sheet_response", return_value=None),
    ):
        assert scan._analyze_contact_sheets(
            frames,
            sheet_dir=tmp_path / "sheets",
            prompt="P",
            system_prompt="S",
            ollama_host="http://x",
            ollama_model="m",
            timeout=5.0,
            client=None,
        ) == ({}, {})


def test_analyze_contact_sheets_requires_full_batch(tmp_path: Path) -> None:
    """A frame count below the slot count produces no sheet at all."""
    frames = [{"frame_path": str(tmp_path / "f0.jpg"), "timestamp": 0.0}]

    with patch.object(scan, "compose_frame_mosaic") as mosaic:
        assert scan._analyze_contact_sheets(
            frames,
            sheet_dir=tmp_path / "sheets",
            prompt="P",
            system_prompt="S",
            ollama_host="http://x",
            ollama_model="m",
            timeout=5.0,
            client=None,
        ) == ({}, {})
    mosaic.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #75 — pre-screen runs before Pass 0 and can drop whole clips
# ---------------------------------------------------------------------------


def _prescreen_order_clip(tmp_path: Path) -> dict:
    video = tmp_path / "ps_clip.mp4"
    video.touch()
    return {
        "clip_id": "ps_clip",
        "file_name": "ps_clip.mp4",
        "file_path": str(video),
        "fps": 25.0,
        "duration": 30.0,
    }


def _prescreen_order_config(gate_mode: str = "enabled", **prescreen: Any) -> dict:
    return {
        "scan_parameters": {
            "domain": "auto",
            "classify_clips": "auto",
            "classify_frames_count": 3,
            "coarse_fps": 0.5,
        },
        "ollama": {"host": "http://localhost:11434", "model": "qwen2.5-vl:7b"},
        "prescreen": {"enabled": True, "gate_mode": gate_mode, **prescreen},
    }


def test_process_clip_prescreen_early_exit_skips_pass0_and_pass1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A clip whose windows are all rejected never reaches Pass 0 or Pass 1."""
    from utils.checkpoint import ClipScanCheckpoint
    from utils.models import ClipClassification, PrescreenRecord
    from utils.prescreen import PrescreenResult

    caplog.set_level(logging.INFO)
    clip = _prescreen_order_clip(tmp_path)
    record = _make_prescreen_record(interval={"start": 0.0, "end": 30.0})
    result = PrescreenResult(records=[record], rejected_intervals=[(0.0, 30.0)], duration=30.0)

    pass0_collector: list[ClipClassification] = []
    prescreen_collector: list[PrescreenRecord] = []
    flush_calls: list[str] = []
    checkpoints_dir = tmp_path / "checkpoints"

    with (
        patch.object(scan, "_prescreen_clip", return_value=(result, None)),
        patch.object(scan, "extract_classification_frames") as mock_classify_frames,
        patch.object(scan, "extract_frames") as mock_extract,
        patch.object(scan, "call_ollama") as mock_call,
    ):
        detections = scan.process_clip(
            clip,
            _prescreen_order_config(),
            tmp_path / "work",
            checkpoints_dir=checkpoints_dir,
            pass0_collector=pass0_collector,
            prescreen_collector=prescreen_collector,
            on_trace_flush=flush_calls.append,
        )

    assert detections == []
    mock_classify_frames.assert_not_called()
    mock_extract.assert_not_called()
    mock_call.assert_not_called()
    assert prescreen_collector == [record]
    assert "prescreen" in flush_calls
    assert "pass0" in flush_calls
    assert len(pass0_collector) == 1
    assert pass0_collector[0].disposition == "skipped_prescreen"
    assert pass0_collector[0].usable is False
    checkpoint = ClipScanCheckpoint.load(checkpoints_dir / "ps_clip.json")
    assert checkpoint is not None
    assert checkpoint.pass_stage == "completed"
    assert checkpoint.detected_domain == "unusable"
    assert "rejected by pre-screen" in caplog.text


def test_process_clip_prescreen_advisory_never_drops_clip(tmp_path: Path) -> None:
    """Advisory mode defers windows and the clip continues into Pass 0."""
    from utils.prescreen import PrescreenResult

    clip = _prescreen_order_clip(tmp_path)
    record = _make_prescreen_record(interval={"start": 0.0, "end": 30.0}, action="defer")
    result = PrescreenResult(records=[record], deferred_intervals=[(0.0, 30.0)], duration=30.0)

    with (
        patch.object(scan, "_prescreen_clip", return_value=(result, None)),
        patch.object(scan, "extract_classification_frames", return_value=[]) as mock_frames,
        patch.object(scan, "extract_frames", return_value=[]),
    ):
        scan.process_clip(clip, _prescreen_order_config("advisory"), tmp_path / "work")

    mock_frames.assert_called_once()


def test_process_clip_prescreen_partial_reject_continues(tmp_path: Path) -> None:
    """Only some rejected windows keep the clip in the pipeline."""
    from utils.prescreen import PrescreenResult

    clip = _prescreen_order_clip(tmp_path)
    records = [
        _make_prescreen_record(interval={"start": 0.0, "end": 10.0}),
        _make_prescreen_record(interval={"start": 10.0, "end": 30.0}, action="keep"),
    ]
    result = PrescreenResult(records=records, rejected_intervals=[(0.0, 10.0)], duration=30.0)

    with (
        patch.object(scan, "_prescreen_clip", return_value=(result, None)),
        patch.object(scan, "extract_classification_frames", return_value=[]) as mock_frames,
        patch.object(scan, "extract_frames", return_value=[]),
    ):
        scan.process_clip(clip, _prescreen_order_config(), tmp_path / "work")

    mock_frames.assert_called_once()


def test_process_clip_prescreen_runs_before_pass0(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The pre-screen log line precedes every Pass 0 and Pass 1 line."""
    from utils.prescreen import PrescreenResult

    caplog.set_level(logging.INFO)
    clip = _prescreen_order_clip(tmp_path)
    order: list[str] = []

    def fake_prescreen(*args, **kwargs):
        order.append("prescreen")
        return PrescreenResult(duration=30.0), None

    def fake_classify_frames(*args, **kwargs):
        order.append("pass0")
        return []

    def fake_extract(*args, **kwargs):
        order.append("pass1")
        return []

    with (
        patch.object(scan, "_prescreen_clip", side_effect=fake_prescreen),
        patch.object(scan, "extract_classification_frames", side_effect=fake_classify_frames),
        patch.object(scan, "extract_frames", side_effect=fake_extract),
    ):
        scan.process_clip(clip, _prescreen_order_config(), tmp_path / "work")

    assert order == ["prescreen", "pass0", "pass1"]
    messages = [r.getMessage() for r in caplog.records]
    first_pass0 = next(i for i, m in enumerate(messages) if m.startswith("[Pass 0]"))
    first_pass1 = next(i for i, m in enumerate(messages) if m.startswith("[Pass 1]"))
    assert first_pass0 < first_pass1


def test_process_clip_contact_sheets_skip_prescreen_rejected_frames(tmp_path: Path) -> None:
    """Frames inside rejected windows are never tiled into contact sheets."""
    from utils.prescreen import PrescreenResult

    clip = _prescreen_order_clip(tmp_path)
    config = _prescreen_order_config(contact_sheet=True)
    config["scan_parameters"].update({"domain": "underwater", "classify_clips": "off"})
    records = [
        _make_prescreen_record(interval={"start": 0.0, "end": 10.0}),
        _make_prescreen_record(interval={"start": 10.5, "end": 30.0}, action="keep"),
    ]
    result = PrescreenResult(records=records, rejected_intervals=[(0.0, 10.0)], duration=30.0)
    frames = [
        {"frame_path": str(tmp_path / f"c{i}.jpg"), "timestamp": float(ts), "frame_number": i}
        for i, ts in enumerate([2, 6, 12, 16, 20, 24])
    ]

    with (
        patch.object(scan, "_prescreen_clip", return_value=(result, None)),
        patch.object(scan, "extract_frames", return_value=frames),
        patch.object(scan, "_analyze_contact_sheets", return_value=({}, {})) as mock_sheets,
        patch.object(scan, "call_ollama", return_value={"detected": False, "confidence": 0.1}),
    ):
        scan.process_clip(clip, config, tmp_path / "work")

    sheet_frames = mock_sheets.call_args.args[0]
    assert [f["timestamp"] for f in sheet_frames] == [12.0, 16.0, 20.0, 24.0]
