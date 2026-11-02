"""Shared Pydantic data models for Dive Clip TUI."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClipManifest(BaseModel):
    """A single clip from the DaVinci Resolve Media Pool."""

    clip_id: str = Field(..., description="Unique clip identifier")
    file_name: str = Field(..., description="Original file name")
    file_path: str = Field(..., description="Absolute file path (after path mapping)")
    fps: str = Field(default="25.0", description="Clip frame rate as string")
    resolution: str = Field(default="1920x1080", description="WidthxHeight")
    tags: list[str] = Field(default_factory=list, description="Resolve keywords")
    domain: str = Field(default="", description="Detected or configured video domain")
    duration: str = Field(default="", description="Clip duration")
    start_timecode: str = Field(default="", description="Starting timecode")
    end_timecode: str = Field(default="", description="Ending timecode")


class ManifestOutput(BaseModel):
    """Top-level manifest output."""

    project_name: str = Field(..., description="Resolve project name")
    total_clips: int = Field(..., ge=0)
    clips: list[ClipManifest] = Field(default_factory=list)


class Detection(BaseModel):
    """A single AI detection from a video frame."""

    timestamp: float = Field(..., ge=0, description="Seconds into the clip")
    subject: str = Field(..., description="Detected subject")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence 0-1")
    quality_score: int = Field(..., ge=1, le=10, description="Quality score 1-10")
    description: str = Field(..., description="Human-readable description")
    frame_number: int = Field(..., ge=0, description="Frame number in source clip")
    template: str = Field(default="", description="Which prompt template matched")
    domain: str = Field(default="", description="Detected or configured video domain")


class RefinedDetection(BaseModel):
    """A refined detection after fine scanning, ready for EDL output."""

    clip_id: str
    file_name: str
    file_path: str
    fps: str = Field(default="25.0", description="Clip frame rate as string")
    start_time: float = Field(..., ge=0, description="Start of highlight window (seconds)")
    end_time: float = Field(..., ge=0, description="End of highlight window (seconds)")
    subject: str
    confidence: float = Field(..., ge=0, le=1)
    quality_score: int = Field(..., ge=1, le=10)
    description: str
    color: str = Field(default="", description="Marker color: Cyan, Yellow, Green")
    template: str = Field(default="")
    domain: str = Field(default="", description="Detected or configured video domain")


class DetectionsRefined(BaseModel):
    """Top-level refined detections output."""

    total_clips: int = Field(..., ge=0)
    total_detections: int = Field(..., ge=0)
    detections: list[RefinedDetection] = Field(default_factory=list)


class ClipClassification(BaseModel):
    """Pass 0 domain classification result for a single video clip."""

    clip_id: str = Field(..., description="Unique clip identifier")
    file_name: str = Field(..., description="Original file name")
    file_path: str = Field(..., description="Absolute file path")
    domain: str = Field(..., description="Classified domain or generic")
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Mean confidence score for the winning domain"
    )
    uncertain: bool = Field(default=False, description="Whether classification is uncertain")
    reason: str | None = Field(default=None, description="Reason for uncertainty, if any")
    usable: bool = Field(
        default=True, description="Whether footage is usable (not in-bag, covered, or dangling)"
    )
    quality_score: int | None = Field(
        default=None, ge=1, le=10, description="Estimated visual quality score (1-10)"
    )
    environment: str = Field(default="", description="Inferred physical setting or environment")
    primary_subjects: list[str] = Field(
        default_factory=list, description="Primary visible subjects or elements"
    )


class ClassificationOutput(BaseModel):
    """Top-level classification output file schema."""

    total_clips: int = Field(..., ge=0)
    classifications: list[ClipClassification] = Field(default_factory=list)
