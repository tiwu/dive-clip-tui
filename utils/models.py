"""Shared Pydantic data models for Dive Clip TUI."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

StoryRole = Literal[
    "establishing",
    "entry_transition",
    "hero_subject",
    "human_moment",
    "unexpected_event",
    "none",
]
VALID_STORY_ROLES: frozenset[str] = frozenset({
    "establishing",
    "entry_transition",
    "hero_subject",
    "human_moment",
    "unexpected_event",
    "none",
})


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
    pix_fmt: str = Field(default="", description="Pixel format (e.g. yuv420p, yuv420p10le)")
    has_audio: bool = Field(default=False, description="Clip contains an audio stream")
    has_telemetry: bool = Field(default=False, description="Clip contains a sensor data stream")
    telemetry_type: str = Field(
        default="", description="Sensor track family: gopro, dji, insta360, or empty"
    )
    telemetry_stream_index: int | None = Field(
        default=None, description="Container stream index carrying telemetry"
    )
    is_high_fps: bool = Field(default=False, description="Frame rate indicates slow-motion intent")
    hilight_tags: list[float] = Field(
        default_factory=list, description="GoPro HiLight marker timestamps in seconds"
    )
    creation_time: str = Field(default="", description="Container creation time tag")
    bit_rate: float | None = Field(default=None, description="Container bit rate in bits/second")
    bitrate_density: float | None = Field(
        default=None, description="Bits per pixel per second (bit_rate / w / h / fps)"
    )


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
    story_role: str = Field(default="hero_subject", description="Story role")
    detection_id: str = Field(default="", description="Deterministic detection identifier")
    sort_key: str = Field(default="", description="Canonical sort key")
    hsv_histogram: list[float] = Field(
        default_factory=list, description="Compact HSV histogram of the highlight frame"
    )
    hsv_entropy: float | None = Field(
        default=None, description="Shannon entropy of the HSV histogram"
    )
    telemetry_score: float | None = Field(
        default=None, description="Telemetry action score in 0..1, None when unavailable"
    )

    @model_validator(mode="after")
    def populate_deterministic_ids(self) -> Self:
        if not self.detection_id:
            self.detection_id = f"{self.clip_id}_{self.start_time:08.2f}_{self.end_time:08.2f}"
        if not self.sort_key:
            self.sort_key = (
                f"{self.file_name}:{self.start_time:08.2f}:{self.end_time:08.2f}:{self.clip_id}"
            )
        return self


class DetectionsRefined(BaseModel):
    """Top-level refined detections output."""

    total_clips: int = Field(..., ge=0)
    total_detections: int = Field(..., ge=0)
    detections: list[RefinedDetection] = Field(default_factory=list)


class Pass0FrameTrace(BaseModel):
    """Trace of a single frame evaluated during Pass 0."""

    timestamp: float = Field(..., ge=0)
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)
    domain: str = ""
    suggested_domain: str | None = None
    confidence: float | None = None
    usable: bool = True
    quality_score: int | None = None
    reject_reasons: list[str] = Field(default_factory=list)
    scene_description: str = ""


class ClipClassification(BaseModel):
    """Pass 0 domain classification result for a single video clip."""

    clip_id: str = Field(..., description="Unique clip identifier")
    file_name: str = Field(..., description="Original file name")
    file_path: str = Field(..., description="Absolute file path")
    domain: str = Field(..., description="Classified domain or generic")
    suggested_domain: str | None = Field(
        default=None, description="Suggested specific domain if classified as generic"
    )
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
    sort_key: str = Field(
        default="", description="Canonical sort key (source_file:start:end:clip_id)"
    )
    disposition: str = Field(
        default="accepted",
        description="accepted | skipped_unusable | skipped_non_target_domain | reused_from_input",
    )
    frame_evaluations: list[Pass0FrameTrace] = Field(
        default_factory=list, description="Trace of Pass 0 frame evaluations"
    )


class ClassificationOutput(BaseModel):
    """Top-level classification output file schema."""

    total_clips: int = Field(..., ge=0)
    classifications: list[ClipClassification] = Field(default_factory=list)


class CoarseFrameTrace(BaseModel):
    """Trace of a single frame evaluated during Pass 1 coarse scan."""

    detection_id: str
    clip_id: str
    source_file: str
    timestamp: float
    start_time: float
    end_time: float
    sort_key: str
    detected: bool
    subject: str = ""
    category: str = ""
    story_role: str = "hero_subject"
    confidence: float = 0.0
    quality_score: int = 0
    reject_reasons: list[str] = Field(default_factory=list)
    description: str = ""
    passed_gate: bool = False
    decision_reason: str = ""
    candidate_window: list[float] | tuple[float, float] | None = None
    prescreen_action: str = Field(default="", description="Pre-VLM screening decision")
    telemetry_score: float | None = Field(
        default=None, description="Telemetry action score in 0..1, None when unavailable"
    )
    hsv_entropy: float | None = Field(
        default=None, description="Shannon entropy of the sampled HSV histogram"
    )
    laplacian_sharpness: float | None = Field(
        default=None, description="Laplacian variance of the sampled frame"
    )
    audio_peak_db: float | None = Field(
        default=None, description="Peak audio level in dB for the sampled window"
    )
    audio_spike_triggered: bool | None = Field(
        default=None, description="Whether the audio peak exceeded the spike threshold"
    )
    contact_sheet_path: str = Field(default="", description="Path to the 2x2 contact sheet")
    tile_index: int | None = Field(
        default=None, description="Stable index of this tile within the contact sheet"
    )
    tile_timestamp: float | None = Field(
        default=None, description="Source timestamp in seconds for this contact-sheet tile"
    )
    tile_count: int | None = Field(
        default=None, description="Number of tiles the contact sheet contained"
    )


class Pass1ClipTrace(BaseModel):
    """Trace of Pass 1 coarse screening for a single clip."""

    clip_id: str
    source_file: str
    sort_key: str
    detected_domain: str
    template_used: str
    coarse_threshold: float
    coarse_quality_threshold: int
    frame_traces: list[CoarseFrameTrace] = Field(default_factory=list)
    merged_windows: list[list[float]] = Field(default_factory=list)
    window_routing: dict[str, str] = Field(default_factory=dict)


class Pass1TraceOutput(BaseModel):
    """Top-level Pass 1 coarse trace file schema."""

    total_clips: int = Field(..., ge=0)
    clips: list[Pass1ClipTrace] = Field(default_factory=list)


class FineFrameTrace(BaseModel):
    """Trace of a single frame evaluated during Pass 2 fine scan."""

    window_index: int = 0
    detection_id: str
    clip_id: str
    source_file: str
    start_time: float
    end_time: float
    sort_key: str
    template: str = ""
    detected: bool
    subject: str = ""
    story_role: str = "hero_subject"
    confidence: float = 0.0
    quality_score: int = 0
    reject_reasons: list[str] = Field(default_factory=list)
    passed_gate: bool = False
    decision_reason: str = ""


class MergeEvent(BaseModel):
    """Record of contiguous detection window merging or duration splitting."""

    merged_detection_id: str
    constituent_ids: list[str] = Field(default_factory=list)
    matched_tokens: list[str] = Field(default_factory=list)
    action: str = "merged"


class DroppedOverlap(BaseModel):
    """Record of an overlapping detection dropped in favor of higher quality."""

    dropped_detection_id: str
    dropped_subject: str
    dropped_quality_score: int
    superseded_by_detection_id: str
    superseded_by_subject: str
    superseded_by_quality_score: int
    reason: str = "lower_quality_or_confidence"


class Pass2ClipTrace(BaseModel):
    """Trace of Pass 2 fine scan and shot merging for a single clip."""

    clip_id: str
    source_file: str
    sort_key: str
    quality_threshold: int
    fine_pre_roll_seconds: float = 1.5
    fine_post_roll_seconds: float = 2.0
    merge_gap_seconds: float = 1.5
    max_clip_duration: float = 12.0
    frame_traces: list[FineFrameTrace] = Field(default_factory=list)
    merge_events: list[MergeEvent] = Field(default_factory=list)
    dropped_overlaps: list[DroppedOverlap] = Field(default_factory=list)


class Pass2TraceOutput(BaseModel):
    """Top-level Pass 2 fine trace file schema."""

    total_clips: int = Field(..., ge=0)
    clips: list[Pass2ClipTrace] = Field(default_factory=list)


class SelectionCandidateTrace(BaseModel):
    """Detailed score and decision breakdown for an MMR candidate."""

    detection_id: str
    clip_id: str
    source_file: str
    start_time: float
    end_time: float
    sort_key: str
    subject: str = ""
    story_role: str = "hero_subject"
    template: str = ""
    quality_score: int = 0
    confidence: float = 0.0
    relevance_score: float = 0.0
    max_similarity_penalty: float = 0.0
    most_similar_selected_id: str | None = None
    quota_penalty: float = 0.0
    mmr_score: float = 0.0
    selected: bool = False
    selection_rank: int | None = None
    chronological_rank: int | None = None
    decision_reason: str = ""
    telemetry_score: float | None = Field(
        default=None, description="Telemetry action score in 0..1, None when unavailable"
    )
    hsv_similarity: float | None = Field(
        default=None, description="HSV histogram similarity against the most similar selection"
    )


class SelectionTraceOutput(BaseModel):
    """Top-level MMR selection trace schema."""

    top_n: int = Field(..., ge=0)
    lambda_quality: float = 0.65
    per_clip_limit: int = 1
    total_candidates: int = Field(..., ge=0)
    selected_count: int = Field(..., ge=0)
    candidates: list[SelectionCandidateTrace] = Field(default_factory=list)


class PrescreenGateOutcome(BaseModel):
    """Outcome of one pre-VLM gate evaluation."""

    name: str
    value: float | None = None
    threshold: float | None = None
    triggered: bool = False
    outcome: Literal["pass", "reject", "advisory"] = "pass"
    reason: str = ""


class PrescreenScoreComponent(BaseModel):
    """One weighted contribution to the telemetry action score."""

    name: str
    raw: float | None = None
    normalized: float | None = None
    weight: float = 0.0
    contribution: float | None = None


class PrescreenInterval(BaseModel):
    """Candidate window a pre-VLM screening record describes."""

    start: float = Field(..., ge=0, description="Window start in seconds")
    end: float = Field(..., ge=0, description="Window end in seconds")
    sample_count: int = Field(default=0, ge=0, description="Telemetry samples in the window")


class PrescreenRecord(BaseModel):
    """One pre-VLM screening decision for a single candidate window."""

    clip_id: str
    source_file: str
    sort_key: str = ""
    candidate_index: int = Field(default=0, ge=0)
    interval: PrescreenInterval
    gate_mode: Literal["advisory", "enabled"] = "advisory"
    is_high_fps: bool = False
    hilight_tags: list[float] = Field(default_factory=list)
    hilight_candidates: list[list[float]] = Field(default_factory=list)
    bitrate_density: float | None = None
    telemetry_type: str = ""
    scen_dominant: str | None = None
    scen_confidence: float | None = None
    scen_coverage: float | None = None
    domain_prior_applied: bool = False
    domain_prior_reason: str = ""
    yavg: float | None = None
    unif: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    tilt_deg: float | None = None
    gyro_rms: float | None = None
    gps_speed_ms: float | None = None
    face_detected: bool | None = None
    audio_peak_db: float | None = None
    audio_spike_triggered: bool = False
    laplacian_sharpness: float | None = None
    hsv_entropy: float | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    gates: list[PrescreenGateOutcome] = Field(default_factory=list)
    telemetry_score: float | None = None
    score_components: list[PrescreenScoreComponent] = Field(default_factory=list)
    action: Literal["keep", "reject", "defer"] = "keep"
    vlm_call: bool = False
    reason: str = ""


class PrescreenTraceOutput(BaseModel):
    """Top-level prescreen trace file schema."""

    version: int = 1
    total_clips: int = Field(default=0, ge=0)
    total_records: int = Field(default=0, ge=0)
    records: list[PrescreenRecord] = Field(default_factory=list)


class PromptCandidate(BaseModel):
    """A generated prompt template candidate for human evaluation; never run by scans."""

    name: str
    domain: str = "generic"
    coarse_prompt: str
    fine_prompt: str
    recommended_threshold: float = Field(0.65, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    color: str = "Magenta"
    status: str = "candidate"
    environment: str = Field("", max_length=120)
    primary_subjects: list[str] = Field(default_factory=list, max_length=10)


class PromptCandidateOutput(BaseModel):
    """Schema of the prompt_candidates.json file written by `dive-clip classify`."""

    name: str
    domain: str = "generic"
    inferred_context: dict[str, Any] = Field(default_factory=dict)
    coarse_prompt: str
    fine_prompt: str
    recommended_threshold: float = Field(0.65, ge=0.0, le=1.0)
    total_candidates: int = Field(..., ge=0)
    candidates: list[PromptCandidate] = Field(default_factory=list)
