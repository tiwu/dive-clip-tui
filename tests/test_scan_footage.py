"""Tests for scripts/02_scan_footage.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        clip_id="c1", file_name="a.mp4", file_path="/a.mp4",
        start_time=0.0, end_time=5.0, subject="a",
        confidence=0.8, quality_score=8, description="a",
    )
    det2 = RefinedDetection(
        clip_id="c1", file_name="a.mp4", file_path="/a.mp4",
        start_time=10.0, end_time=15.0, subject="b",
        confidence=0.9, quality_score=9, description="b",
    )
    result = scan._deduplicate_detections([det1, det2])
    assert len(result) == 2


def test_deduplicate_detections_overlap_keeps_higher_quality() -> None:
    """_deduplicate_detections should keep higher quality score when overlapping."""
    from utils.models import RefinedDetection

    det_low = RefinedDetection(
        clip_id="c1", file_name="a.mp4", file_path="/a.mp4",
        start_time=0.0, end_time=10.0, subject="a",
        confidence=0.8, quality_score=5, description="low",
    )
    det_high = RefinedDetection(
        clip_id="c1", file_name="a.mp4", file_path="/a.mp4",
        start_time=5.0, end_time=15.0, subject="b",
        confidence=0.9, quality_score=9, description="high",
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

    with patch.object(scan.requests, "post", return_value=mock_response):
        result = scan.call_ollama("http://localhost:11434", "qwen2.5-vl:7b", str(image_path), "prompt")

    assert result is not None
    assert result["detected"] is True
    assert result["subject"] == "turtle"


def test_call_ollama_request_failure(tmp_path: Path) -> None:
    """call_ollama should return None when request fails."""
    import requests

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")

    with patch.object(
        scan.requests, "post", side_effect=requests.exceptions.RequestException("connection error")
    ):
        result = scan.call_ollama("http://localhost:11434", "qwen2.5-vl:7b", str(image_path), "prompt")

    assert result is None


def test_call_ollama_invalid_json(tmp_path: Path) -> None:
    """call_ollama should return None when response is invalid JSON."""
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "not json"}

    with patch.object(scan.requests, "post", return_value=mock_response):
        result = scan.call_ollama("http://localhost:11434", "qwen2.5-vl:7b", str(image_path), "prompt")

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


def test_process_clip_with_candidates(tmp_path: Path) -> None:
    """process_clip should run coarse-to-fine scan and return detections."""
    video = tmp_path / "video.mp4"
    video.touch()

    clip = {
        "file_path": str(video),
        "file_name": "video.mp4",
        "clip_id": "c1",
        "fps": "25.0",
        "tags": ["turtle"],
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
        {"frame_path": str(tmp_path / "fine_frame_000001.jpg"), "timestamp": 3.0, "frame_number": 1},
    ]
    (tmp_path / "fine_frame_000001.jpg").touch()

    def mock_extract_frames(video_path, output_dir, fps=0.5, scale=None, start_seconds=None, duration_seconds=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if start_seconds is not None:
            (output_dir / "fine_frame_000001.jpg").touch()
            return fine_frames
        else:
            (output_dir / "frame_000001.jpg").touch()
            return coarse_frames

    mock_ollama_result = {
        "detected": True,
        "subject": "turtle",
        "confidence": 0.8,
        "quality_score": 8,
        "description": "Sea turtle",
    }

    with patch.object(scan, "extract_frames", side_effect=mock_extract_frames):
        with patch.object(scan, "call_ollama", return_value=mock_ollama_result):
            result = scan.process_clip(clip, config, tmp_path / "work")

    assert len(result) > 0
    assert result[0].subject == "turtle"
    assert result[0].quality_score == 8
