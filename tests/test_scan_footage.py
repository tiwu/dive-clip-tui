"""Tests for scripts/02_scan_footage.py."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
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
        "[Pass 1] Frame 1 @ 2.0s: detected=True confidence=0.80 passes_threshold=True"
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

    # Should have used broad marine template since tags are empty
    assert len(captured_prompts) >= 1
    prompt_used = captured_prompts[0]
    assert BROAD_MARINE_TEMPLATE.user_prompt in prompt_used, (
        f"Expected broad marine template prompt in coarse pass, got: {prompt_used[:80]}..."
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
    assert BROAD_MARINE_TEMPLATE.user_prompt in fine_prompt, (
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
    assert PELAGICS_TEMPLATE.user_prompt in prompt_used, (
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
    assert MACRO_TEMPLATE.user_prompt in prompt_used, (
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
        patch.object(scan, "OllamaClient") as client_cls,
        patch.object(scan, "extract_frames", return_value=[]),
    ):
        scan.process_clip(clip, config, tmp_path / "work")

    assert client_cls.call_args.args[0] == "http://plotter.local:11434"
