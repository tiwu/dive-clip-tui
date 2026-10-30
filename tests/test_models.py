"""Tests for utils/models.py."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from utils.models import (
    ClipManifest,
    Detection,
    DetectionsRefined,
    ManifestOutput,
    RefinedDetection,
)


def test_clip_manifest_minimal() -> None:
    """ClipManifest should work with only required fields."""
    clip = ClipManifest(
        clip_id="abc123",
        file_name="dive.mp4",
        file_path="/tmp/dive.mp4",
    )
    assert clip.clip_id == "abc123"
    assert clip.file_name == "dive.mp4"
    assert clip.file_path == "/tmp/dive.mp4"
    assert clip.fps == "25.0"  # default
    assert clip.resolution == "1920x1080"  # default
    assert clip.tags == []  # default
    assert clip.duration == ""  # default


def test_clip_manifest_full() -> None:
    """ClipManifest should accept all fields."""
    clip = ClipManifest(
        clip_id="xyz789",
        file_name="shark.mp4",
        file_path="/mnt/videos/shark.mp4",
        fps="30.0",
        resolution="3840x2160",
        tags=["shark", "pelagic"],
        duration="120.5",
        start_timecode="00:00:00:00",
        end_timecode="00:02:00:00",
    )
    assert clip.fps == "30.0"
    assert clip.resolution == "3840x2160"
    assert clip.tags == ["shark", "pelagic"]
    assert clip.duration == "120.5"


def test_manifest_output() -> None:
    """ManifestOutput should wrap a list of clips."""
    clips = [
        ClipManifest(clip_id="c1", file_name="a.mp4", file_path="/a.mp4"),
        ClipManifest(clip_id="c2", file_name="b.mp4", file_path="/b.mp4"),
    ]
    output = ManifestOutput(
        project_name="Test Project",
        total_clips=2,
        clips=clips,
    )
    assert output.project_name == "Test Project"
    assert output.total_clips == 2
    assert len(output.clips) == 2


def test_detection_valid() -> None:
    """Detection should accept valid field values."""
    det = Detection(
        timestamp=10.5,
        subject="turtle",
        confidence=0.95,
        quality_score=8,
        description="Sea turtle swimming",
        frame_number=250,
        template="pelagics",
    )
    assert det.timestamp == 10.5
    assert det.subject == "turtle"
    assert det.confidence == 0.95
    assert det.quality_score == 8
    assert det.frame_number == 250


def test_detection_confidence_out_of_range() -> None:
    """Detection should reject confidence outside [0, 1]."""
    with pytest.raises(ValidationError):
        Detection(
            timestamp=10.0,
            subject="turtle",
            confidence=1.5,  # > 1
            quality_score=8,
            description="test",
            frame_number=0,
        )


def test_detection_confidence_negative() -> None:
    """Detection should reject negative confidence."""
    with pytest.raises(ValidationError):
        Detection(
            timestamp=10.0,
            subject="turtle",
            confidence=-0.1,  # < 0
            quality_score=8,
            description="test",
            frame_number=0,
        )


def test_detection_quality_score_out_of_range() -> None:
    """Detection should reject quality_score outside [1, 10]."""
    with pytest.raises(ValidationError):
        Detection(
            timestamp=10.0,
            subject="turtle",
            confidence=0.5,
            quality_score=11,  # > 10
            description="test",
            frame_number=0,
        )


def test_detection_quality_score_zero() -> None:
    """Detection should reject quality_score of 0."""
    with pytest.raises(ValidationError):
        Detection(
            timestamp=10.0,
            subject="turtle",
            confidence=0.5,
            quality_score=0,  # < 1
            description="test",
            frame_number=0,
        )


def test_detection_negative_timestamp() -> None:
    """Detection should reject negative timestamp."""
    with pytest.raises(ValidationError):
        Detection(
            timestamp=-1.0,  # < 0
            subject="turtle",
            confidence=0.5,
            quality_score=8,
            description="test",
            frame_number=0,
        )


def test_refined_detection() -> None:
    """RefinedDetection should accept all fields."""
    det = RefinedDetection(
        clip_id="c1",
        file_name="dive.mp4",
        file_path="/tmp/dive.mp4",
        fps="25.0",
        start_time=10.0,
        end_time=15.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="Sea turtle",
        color="Cyan",
        template="pelagics",
    )
    assert det.start_time == 10.0
    assert det.end_time == 15.0
    assert det.color == "Cyan"


def test_detections_refined() -> None:
    """DetectionsRefined should wrap a list of refined detections."""
    det1 = RefinedDetection(
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
    det2 = RefinedDetection(
        clip_id="c2",
        file_name="b.mp4",
        file_path="/b.mp4",
        start_time=5.0,
        end_time=10.0,
        subject="shark",
        confidence=0.8,
        quality_score=9,
        description="test",
    )
    output = DetectionsRefined(
        total_clips=2,
        total_detections=2,
        detections=[det1, det2],
    )
    assert output.total_clips == 2
    assert output.total_detections == 2
    assert len(output.detections) == 2


def test_model_serialization() -> None:
    """Models should serialize to dict via model_dump()."""
    clip = ClipManifest(
        clip_id="c1",
        file_name="test.mp4",
        file_path="/tmp/test.mp4",
    )
    data = clip.model_dump()
    assert data["clip_id"] == "c1"
    assert data["file_name"] == "test.mp4"
    assert data["file_path"] == "/tmp/test.mp4"


def test_model_from_dict() -> None:
    """Models should be constructable from dict."""
    data = {
        "clip_id": "c1",
        "file_name": "test.mp4",
        "file_path": "/tmp/test.mp4",
        "tags": ["turtle"],
    }
    clip = ClipManifest(**data)
    assert clip.tags == ["turtle"]
