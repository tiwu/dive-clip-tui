"""Tests for utils/models.py."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from utils.models import (
    ClassificationOutput,
    ClipClassification,
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
    assert clip.pix_fmt == ""  # default


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
        pix_fmt="yuv420p10le",
    )
    assert clip.fps == "30.0"
    assert clip.pix_fmt == "yuv420p10le"
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


def test_clip_classification_valid() -> None:
    """ClipClassification should accept valid fields."""
    c = ClipClassification(
        clip_id="clip_1",
        file_name="clip_1.mp4",
        file_path="/videos/clip_1.mp4",
        domain="underwater",
        confidence=0.88,
        uncertain=False,
    )
    assert c.clip_id == "clip_1"
    assert c.domain == "underwater"
    assert c.confidence == 0.88
    assert c.uncertain is False
    assert c.reason is None


def test_clip_classification_uncertain() -> None:
    """ClipClassification should support uncertain classifications with reason and None confidence."""
    c = ClipClassification(
        clip_id="clip_2",
        file_name="clip_2.mp4",
        file_path="/videos/clip_2.mp4",
        domain="generic",
        confidence=None,
        uncertain=True,
        reason="no usable responses",
    )
    assert c.domain == "generic"
    assert c.confidence is None
    assert c.uncertain is True
    assert c.reason == "no usable responses"


def test_clip_classification_validation_error() -> None:
    """ClipClassification should reject invalid fields."""
    with pytest.raises(ValidationError):
        # Missing required fields
        ClipClassification(file_name="clip.mp4")

    with pytest.raises(ValidationError):
        # Confidence out of range
        ClipClassification(
            clip_id="c1",
            file_name="c.mp4",
            file_path="/c.mp4",
            domain="generic",
            confidence=1.5,
        )


def test_classification_output() -> None:
    """ClassificationOutput should wrap a list of ClipClassification records."""
    c1 = ClipClassification(
        clip_id="c1",
        file_name="a.mp4",
        file_path="/a.mp4",
        domain="motorcycle",
        confidence=0.92,
    )
    output = ClassificationOutput(total_clips=1, classifications=[c1])
    assert output.total_clips == 1
    assert len(output.classifications) == 1
    dumped = json.loads(output.model_dump_json())
    assert dumped["total_clips"] == 1
    assert dumped["classifications"][0]["domain"] == "motorcycle"


def test_clip_classification_context_fields() -> None:
    """ClipClassification should preserve environment and primary_subjects."""
    c_default = ClipClassification(
        clip_id="c_def",
        file_name="clip.mp4",
        file_path="/videos/clip.mp4",
        domain="generic",
    )
    assert c_default.environment == ""
    assert c_default.primary_subjects == []

    c = ClipClassification(
        clip_id="c_generic",
        file_name="alpine.mp4",
        file_path="/videos/alpine.mp4",
        domain="generic",
        confidence=0.82,
        environment="alpine meadow",
        primary_subjects=["marmot", "pine tree"],
    )
    assert c.environment == "alpine meadow"
    assert c.primary_subjects == ["marmot", "pine tree"]
    dumped = json.loads(c.model_dump_json())
    assert dumped["environment"] == "alpine meadow"
    assert dumped["primary_subjects"] == ["marmot", "pine tree"]
