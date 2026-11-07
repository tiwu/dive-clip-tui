"""Pytest configuration and shared fixtures for dive-clip-ai tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import utils.logging_setup  # noqa: F401


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_detections() -> list[dict]:
    """Return a list of sample detection dicts."""
    return [
        {
            "clip_id": "clip001",
            "file_name": "dive1.mp4",
            "file_path": "/tmp/dive1.mp4",
            "fps": "25.0",
            "start_time": 10.0,
            "end_time": 15.0,
            "subject": "turtle",
            "confidence": 0.9,
            "quality_score": 8,
            "description": "Sea turtle swimming",
            "color": "Cyan",
            "template": "pelagics",
        },
        {
            "clip_id": "clip002",
            "file_name": "dive2.mp4",
            "file_path": "/tmp/dive2.mp4",
            "fps": "25.0",
            "start_time": 20.0,
            "end_time": 25.0,
            "subject": "shark",
            "confidence": 0.85,
            "quality_score": 9,
            "description": "Shark passing by",
            "color": "Cyan",
            "template": "pelagics",
        },
        {
            "clip_id": "clip003",
            "file_name": "dive1.mp4",
            "file_path": "/tmp/dive1.mp4",
            "fps": "25.0",
            "start_time": 30.0,
            "end_time": 35.0,
            "subject": "nudibranch",
            "confidence": 0.75,
            "quality_score": 7,
            "description": "Colorful nudibranch",
            "color": "Yellow",
            "template": "macro",
        },
    ]


@pytest.fixture
def sample_detections_file(tmp_path: Path, sample_detections: list[dict]) -> Path:
    """Write sample detections to a JSON file and return its path."""
    data = {
        "detections": sample_detections,
        "total_detections": len(sample_detections),
        "total_clips": 2,
    }
    p = tmp_path / "detections_refined.json"
    p.write_text(json.dumps(data, indent=2))
    return p


@pytest.fixture
def sample_manifest(tmp_path: Path) -> Path:
    """Write a sample manifest to a JSON file and return its path."""
    data = {
        "project_name": "Test",
        "total_clips": 2,
        "clips": [
            {
                "clip_id": "clip001",
                "file_name": "dive1.mp4",
                "file_path": "/tmp/dive1.mp4",
                "fps": "25.0",
                "resolution": "1920x1080",
                "tags": ["turtle"],
                "duration": "60.0",
            },
            {
                "clip_id": "clip002",
                "file_name": "dive2.mp4",
                "file_path": "/tmp/dive2.mp4",
                "fps": "25.0",
                "resolution": "1920x1080",
                "tags": ["shark"],
                "duration": "60.0",
            },
        ],
    }
    p = tmp_path / "manifest_raw.json"
    p.write_text(json.dumps(data, indent=2))
    return p
