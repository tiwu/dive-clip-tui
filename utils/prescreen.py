"""Pre-VLM screening funnel: cheap telemetry, visual, and audio gates.

The funnel exists to keep expensive VLM calls away from candidate windows that
cannot contain anything usable (lens cap on, fogged housing, pure shake, steady
ambient audio only). Every decision is recorded as a :class:`PrescreenRecord` so
the screen is auditable after the fact.

Gates run in ``advisory`` mode by default: they are evaluated and recorded, but a
triggered gate never removes a candidate. ``enabled`` mode turns the identical
evaluation into an actual reject. That keeps the funnel measurable before it is
trusted.

Visual metrics are computed in pure Python from ``rgb24`` raw frames pulled
through one FFmpeg pipe, so the module carries no numpy/Pillow dependency.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.ffmpeg_streamer import (
    extract_audio_peak_levels,
    extract_raw_frames,
    probe_duration,
)
from utils.models import (
    PrescreenGateOutcome,
    PrescreenInterval,
    PrescreenRecord,
    PrescreenScoreComponent,
)
from utils.telemetry import (
    DEFAULT_GATE_THRESHOLDS,
    DEFAULT_TELEMETRY_WEIGHTS,
    TelemetryTrack,
    aggregate_window,
    compute_telemetry_score,
    evaluate_gates,
    telemetry_domain_prior,
)

logger = logging.getLogger(__name__)

# Rec. 601 luma weights, shifted by 8 bits so the conversion stays integer-only.
LUMA_RED = 77
LUMA_GREEN = 150
LUMA_BLUE = 29

HISTOGRAM_BINS = 8
# Histogram intersection above this value counts as "same shot".
HUE_CHANGE_SIMILARITY = 0.85
# Sliding baseline width (in samples) for audio spike detection.
DEFAULT_AUDIO_WINDOW = 5
# A sample must beat the rolling median by this many dB to count as a spike.
DEFAULT_AUDIO_SPIKE_DB = 10.0
# Absolute floor: quiet steady audio must never read as a spike.
DEFAULT_AUDIO_SPIKE_FLOOR_DB = -35.0
# Median Laplacian variance below this reads as out-of-focus / fogged footage.
DEFAULT_BLUR_MIN = 12.0
DEFAULT_SAMPLE_SIZE = 64
DEFAULT_SAMPLE_FPS = 0.5
# Luma standard deviation that maps to maximal non-uniformity (0..255 scale).
UNIFORMITY_STDEV_SCALE = 64.0
DEFAULT_HILIGHT_PAD_SECONDS = 4.0
VALID_GATE_MODES = ("advisory", "enabled")


@dataclass(frozen=True)
class PrescreenConfig:
    """Configuration for the pre-VLM screening funnel."""

    enabled: bool = False
    gate_mode: str = "advisory"
    sample_fps: float = DEFAULT_SAMPLE_FPS
    sample_size: int = DEFAULT_SAMPLE_SIZE
    keyframes_only: bool = True
    audio_enabled: bool = True
    audio_spike_db: float = DEFAULT_AUDIO_SPIKE_DB
    audio_spike_floor_db: float = DEFAULT_AUDIO_SPIKE_FLOOR_DB
    audio_window: int = DEFAULT_AUDIO_WINDOW
    blur_min: float = DEFAULT_BLUR_MIN
    contact_sheet: bool = False
    gate_thresholds: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    domain_prior_thresholds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> PrescreenConfig:
        """Build a config from the ``prescreen`` section of ``config.yaml``."""
        raw = (config or {}).get("prescreen") or {}
        if not isinstance(raw, dict):
            logger.warning("prescreen config is not a mapping; using defaults")
            raw = {}
        mode = str(raw.get("gate_mode", "advisory")).strip().lower()
        if mode not in VALID_GATE_MODES:
            logger.warning("Unknown prescreen gate_mode %r; using 'advisory'", mode)
            mode = "advisory"
        return cls(
            enabled=_as_bool(raw.get("enabled"), False),
            gate_mode=mode,
            sample_fps=_as_float(raw.get("sample_fps"), DEFAULT_SAMPLE_FPS),
            sample_size=_as_int(raw.get("sample_size"), DEFAULT_SAMPLE_SIZE),
            keyframes_only=_as_bool(raw.get("keyframes_only"), True),
            audio_enabled=_as_bool(raw.get("audio_enabled"), True),
            audio_spike_db=_as_float(raw.get("audio_spike_db"), DEFAULT_AUDIO_SPIKE_DB),
            audio_spike_floor_db=_as_float(
                raw.get("audio_spike_floor_db"), DEFAULT_AUDIO_SPIKE_FLOOR_DB
            ),
            audio_window=_as_int(raw.get("audio_window"), DEFAULT_AUDIO_WINDOW),
            blur_min=_as_float(raw.get("blur_min"), DEFAULT_BLUR_MIN),
            contact_sheet=_as_bool(raw.get("contact_sheet"), False),
            gate_thresholds=_as_float_map(raw.get("gate_thresholds")),
            weights=_as_float_map(raw.get("weights")),
            domain_prior_thresholds=_as_float_map(raw.get("domain_prior_thresholds")),
        )

    def telemetry_gate_thresholds(self) -> dict[str, float]:
        """Return telemetry gate thresholds with config overrides applied."""
        return {**DEFAULT_GATE_THRESHOLDS, **self.gate_thresholds}

    def telemetry_weights(self) -> dict[str, float]:
        """Return score weights with config overrides applied."""
        return {**DEFAULT_TELEMETRY_WEIGHTS, **self.weights}

    def flat_thresholds(self) -> dict[str, float]:
        """Return every numeric threshold applied, for the audit record."""
        thresholds = dict(self.telemetry_gate_thresholds())
        thresholds["blur_min"] = self.blur_min
        thresholds["audio_spike_db"] = self.audio_spike_db
        thresholds["audio_spike_floor_db"] = self.audio_spike_floor_db
        return thresholds


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce YAML-ish values to bool without raising on junk."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, int | float):
        return bool(value)
    return default


def _as_float(value: Any, default: float) -> float:
    """Coerce YAML-ish values to float, falling back to ``default``."""
    try:
        return float(value)
    except TypeError, ValueError:
        return default


def _as_int(value: Any, default: int) -> int:
    """Coerce YAML-ish values to int, falling back to ``default``."""
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _as_float_map(value: Any) -> dict[str, float]:
    """Keep only the numeric entries of a mapping."""
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, float] = {}
    for key, item in value.items():
        try:
            cleaned[str(key)] = float(item)
        except TypeError, ValueError:
            continue
    return cleaned


def candidate_windows(
    tags: Sequence[float], duration: float, pad: float = DEFAULT_HILIGHT_PAD_SECONDS
) -> list[tuple[float, float]]:
    """Promote HiLight markers into candidate windows, clamped to the clip duration."""
    windows: list[tuple[float, float]] = []
    for tag in tags:
        start = max(0.0, float(tag) - pad)
        end = float(tag) + pad
        if duration > 0:
            end = min(end, duration)
        if end <= start:
            continue
        windows.append((round(start, 3), round(end, 3)))
    return windows


def rgb_to_luma(frame: bytes, size: int) -> list[int]:
    """Convert packed ``rgb24`` bytes to 8-bit luma samples."""
    pixel_count = size * size
    usable = min(len(frame) // 3, pixel_count)
    luma: list[int] = []
    for index in range(0, usable * 3, 3):
        red = frame[index]
        green = frame[index + 1]
        blue = frame[index + 2]
        luma.append((red * LUMA_RED + green * LUMA_GREEN + blue * LUMA_BLUE) >> 8)
    return luma


def mean_luma(luma: Sequence[int]) -> float | None:
    """Return the arithmetic mean of luma samples."""
    if not luma:
        return None
    return round(statistics.fmean(luma), 3)


def luma_uniformity(luma: Sequence[int]) -> float | None:
    """Return 0..1 uniformity, where 1 means a completely flat frame.

    Defined as ``1 - stdev / 64``: a dense fog or a lens cap produces a nearly
    constant frame and therefore a uniformity close to 1.
    """
    if len(luma) < 2:
        return None
    spread = statistics.pstdev(luma)
    return round(max(0.0, 1.0 - min(1.0, spread / UNIFORMITY_STDEV_SCALE)), 4)


def laplacian_variance(luma: Sequence[int], size: int) -> float | None:
    """Return the variance of the 4-neighbour Laplacian over interior pixels.

    Low variance means no high-frequency edge survives, which is what out-of-focus
    or fogged footage looks like.
    """
    if size < 3 or len(luma) < size * size:
        return None
    responses: list[float] = []
    for row in range(1, size - 1):
        base = row * size
        for column in range(1, size - 1):
            center = luma[base + column]
            neighbours = (
                luma[base + column - 1]
                + luma[base + column + 1]
                + luma[base - size + column]
                + luma[base + size + column]
            )
            responses.append(float(4 * center - neighbours))
    if not responses:
        return None
    return round(statistics.pvariance(responses), 4)


def hsv_histogram(frame: bytes, size: int, bins: int = HISTOGRAM_BINS) -> list[float]:
    """Return a saturation-weighted hue histogram normalized to sum 1.

    Pure black or fully desaturated frames carry no hue information and return an
    empty histogram, which callers treat as "no similarity signal".
    """
    if bins <= 0:
        return []
    pixel_count = size * size
    usable = min(len(frame) // 3, pixel_count)
    counts = [0.0] * bins
    for index in range(0, usable * 3, 3):
        red = frame[index] / 255.0
        green = frame[index + 1] / 255.0
        blue = frame[index + 2] / 255.0
        high = max(red, green, blue)
        low = min(red, green, blue)
        delta = high - low
        if delta <= 0.0 or high <= 0.0:
            continue
        if high == red:
            hue = (60.0 * ((green - blue) / delta)) % 360.0
        elif high == green:
            hue = 60.0 * ((blue - red) / delta) + 120.0
        else:
            hue = 60.0 * ((red - green) / delta) + 240.0
        saturation = delta / high
        bucket = int(hue // (360.0 / bins)) % bins
        counts[bucket] += saturation * high

    total = sum(counts)
    if total <= 0:
        return []
    return [round(value / total, 4) for value in counts]


def hsv_entropy(histogram: Sequence[float]) -> float | None:
    """Return normalized Shannon entropy (0..1) of a hue histogram."""
    values = [value for value in histogram if value > 0]
    if len(values) < 2:
        return 0.0 if values else None
    entropy = -sum(value * math.log2(value) for value in values)
    ceiling = math.log2(len(histogram))
    if ceiling <= 0:
        return None
    return round(min(1.0, entropy / ceiling), 4)


def hsv_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return histogram-intersection similarity (0..1), or ``None`` if unusable."""
    if not left or not right or len(left) != len(right):
        return None
    return round(sum(min(a, b) for a, b in zip(left, right, strict=True)), 4)


def hue_change_rate(
    histograms: Sequence[Sequence[float]],
    timestamps: Sequence[float],
    similarity: float = HUE_CHANGE_SIMILARITY,
) -> dict[str, Any]:
    """Count shot-scale hue changes and express them per second."""
    transitions = 0
    pairs = 0
    for index in range(1, len(histograms)):
        score = hsv_similarity(histograms[index - 1], histograms[index])
        if score is None:
            continue
        pairs += 1
        if score < similarity:
            transitions += 1
    span = 0.0
    if len(timestamps) >= 2:
        span = round(float(timestamps[-1]) - float(timestamps[0]), 3)
    rate = None
    if pairs and span > 0:
        rate = round(transitions / span, 4)
    return {"hue_changes": transitions, "hue_changes_per_s": rate, "compared_pairs": pairs}


def visual_metrics(
    frames: Sequence[bytes],
    timestamps: Sequence[float] | None = None,
    size: int = DEFAULT_SAMPLE_SIZE,
    bins: int = HISTOGRAM_BINS,
) -> dict[str, Any]:
    """Summarize sampled frames into window-level visual signals.

    Per-frame values are averaged across the window so a single odd frame cannot
    reject a window on its own; the same reason telemetry aggregates to one row
    per candidate window.
    """
    metrics: dict[str, Any] = {
        "frame_count": 0,
        "mean_luma": None,
        "luma_uniformity": None,
        "laplacian_sharpness": None,
        "hsv_entropy": None,
        "hue_changes": 0,
        "hue_changes_per_s": None,
        "histograms": [],
        "timestamps": [],
    }
    lumas: list[float] = []
    uniformities: list[float] = []
    sharpness: list[float] = []
    entropies: list[float] = []
    histograms: list[list[float]] = []
    kept_timestamps: list[float] = []

    for index, frame in enumerate(frames):
        luma = rgb_to_luma(frame, size)
        if not luma:
            continue
        lumas.append(float(mean_luma(luma) or 0.0))
        spread = luma_uniformity(luma)
        if spread is not None:
            uniformities.append(spread)
        sharp = laplacian_variance(luma, size)
        if sharp is not None:
            sharpness.append(sharp)
        histogram = hsv_histogram(frame, size, bins)
        histograms.append(histogram)
        entropy = hsv_entropy(histogram)
        if entropy is not None:
            entropies.append(entropy)
        if timestamps is not None and index < len(timestamps):
            kept_timestamps.append(round(float(timestamps[index]), 3))
        else:
            kept_timestamps.append(float(index))

    if not lumas:
        return metrics

    metrics["frame_count"] = len(lumas)
    metrics["mean_luma"] = round(statistics.fmean(lumas), 3)
    if uniformities:
        metrics["luma_uniformity"] = round(statistics.fmean(uniformities), 4)
    if sharpness:
        metrics["laplacian_sharpness"] = round(statistics.fmean(sharpness), 4)
    if entropies:
        metrics["hsv_entropy"] = round(statistics.fmean(entropies), 4)
    metrics["histograms"] = histograms
    metrics["timestamps"] = kept_timestamps
    metrics.update(hue_change_rate(histograms, kept_timestamps))
    return metrics


def detect_audio_spikes(
    peaks: Sequence[tuple[float, float]],
    spike_db: float = DEFAULT_AUDIO_SPIKE_DB,
    floor_db: float = DEFAULT_AUDIO_SPIKE_FLOOR_DB,
    window: int = DEFAULT_AUDIO_WINDOW,
) -> list[dict[str, float]]:
    """Return audio peaks that jump above their rolling median baseline.

    A steady ambient track (underwater flow noise, engine hum) keeps its median
    and never spikes, and anything below ``floor_db`` is ignored outright so a
    quiet recording cannot produce spikes out of near-silence.
    """
    ordered = sorted(peaks, key=lambda item: item[0])
    levels = [float(level) for _, level in ordered]
    spikes: list[dict[str, float]] = []
    for index, (timestamp, level) in enumerate(ordered):
        if index == 0:
            continue
        history = levels[max(0, index - max(1, window)) : index]
        if not history:
            continue
        baseline = statistics.median(history)
        if float(level) < floor_db:
            continue
        delta = float(level) - baseline
        if delta < spike_db:
            continue
        spikes.append({
            "timestamp": round(float(timestamp), 3),
            "level_db": round(float(level), 2),
            "baseline_db": round(baseline, 2),
            "delta_db": round(delta, 2),
        })
    return spikes


def audio_window_metrics(
    peaks: Sequence[tuple[float, float]],
    start: float,
    end: float,
    spike_db: float = DEFAULT_AUDIO_SPIKE_DB,
    floor_db: float = DEFAULT_AUDIO_SPIKE_FLOOR_DB,
    window: int = DEFAULT_AUDIO_WINDOW,
) -> dict[str, Any]:
    """Collapse audio peaks to the signals one candidate window records."""
    metrics: dict[str, Any] = {
        "audio_peak_db": None,
        "audio_spike_triggered": False,
        "audio_spike_time": None,
        "audio_spike_count": 0,
        "audio_spike_delta_db": None,
    }
    if not peaks:
        return metrics

    in_window = [
        (float(timestamp), float(level))
        for timestamp, level in peaks
        if start <= float(timestamp) <= end
    ]
    if in_window:
        metrics["audio_peak_db"] = round(max(level for _, level in in_window), 2)

    spikes = [
        spike
        for spike in detect_audio_spikes(peaks, spike_db=spike_db, floor_db=floor_db, window=window)
        if start <= spike["timestamp"] <= end
    ]
    if spikes:
        metrics["audio_spike_triggered"] = True
        metrics["audio_spike_count"] = len(spikes)
        metrics["audio_spike_time"] = spikes[0]["timestamp"]
        metrics["audio_spike_delta_db"] = max(spike["delta_db"] for spike in spikes)
    return metrics


def _blur_gate(value: Any, threshold: float, mode: str) -> dict[str, Any]:
    """Build a blur-gate record shaped like :func:`evaluate_gates` output."""
    if not isinstance(value, int | float):
        return {
            "gate": "blur",
            "input": None,
            "threshold": threshold,
            "direction": "below",
            "outcome": "skip",
            "reason": "signal unavailable",
            "mode": mode,
        }
    triggered = float(value) < threshold
    return {
        "gate": "blur",
        "input": round(float(value), 3),
        "threshold": threshold,
        "direction": "below",
        "outcome": "fail" if triggered else "pass",
        "reason": "median Laplacian variance below focus floor"
        if triggered
        else "within threshold",
        "mode": mode,
    }


def build_gates(signals: dict[str, Any], config: PrescreenConfig) -> list[PrescreenGateOutcome]:
    """Evaluate telemetry and visual gates into auditable outcomes."""
    raw_gates = evaluate_gates(
        signals,
        thresholds=config.telemetry_gate_thresholds(),
        mode=config.gate_mode,
    )
    raw_gates.append(
        _blur_gate(signals.get("laplacian_sharpness"), config.blur_min, config.gate_mode)
    )

    outcomes: list[PrescreenGateOutcome] = []
    for gate in raw_gates:
        triggered = gate["outcome"] == "fail"
        if not triggered:
            outcome = "pass"
        elif gate.get("mode") == "enabled":
            outcome = "reject"
        else:
            outcome = "advisory"
        outcomes.append(
            PrescreenGateOutcome(
                name=str(gate["gate"]),
                value=gate.get("input"),
                threshold=gate.get("threshold"),
                triggered=triggered,
                outcome=outcome,
                reason=str(gate.get("reason", "")),
            )
        )
    return outcomes


def resolve_action(gates: Sequence[PrescreenGateOutcome], gate_mode: str) -> tuple[str, list[str]]:
    """Map triggered gates to a screening action and its reasons."""
    triggered = [gate for gate in gates if gate.triggered]
    if not triggered:
        return "keep", []
    reasons = [f"{gate.name}: {gate.reason}" for gate in triggered]
    return ("reject" if gate_mode == "enabled" else "defer"), reasons


def _component_raw(name: str, signals: dict[str, Any]) -> float | None:
    """Return the raw signal behind a score component, for the breakdown."""
    if name == "lean_speed":
        speed = signals.get("gps_speed_ms")
        if isinstance(speed, int | float):
            return round(float(speed), 3)
        leanish = [
            abs(float(signals[key]))
            for key in ("roll_deg", "pitch_deg")
            if isinstance(signals.get(key), int | float)
        ]
        return round(max(leanish), 3) if leanish else None
    key = {
        "gyro_stability": "gyro_rms",
        "face": "face_detected",
        "hue_change": "hue_changes_per_s",
    }.get(name)
    if key is None:
        return None
    value = signals.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None


def score_components(
    score_info: dict[str, Any], signals: dict[str, Any]
) -> list[PrescreenScoreComponent]:
    """Expand a score breakdown into trace rows."""
    components: list[PrescreenScoreComponent] = []
    raw_values: dict[str, float] = score_info.get("components") or {}
    weights: dict[str, float] = score_info.get("weights") or {}
    for name, normalized in raw_values.items():
        weight = float(weights.get(name, 0.0))
        components.append(
            PrescreenScoreComponent(
                name=name,
                raw=_component_raw(name, signals),
                normalized=round(float(normalized), 4),
                weight=round(weight, 4),
                contribution=round(float(normalized) * weight, 4),
            )
        )
    return components


def build_prescreen_record(
    *,
    clip_id: str,
    source_file: str,
    candidate_index: int,
    start: float,
    end: float,
    config: PrescreenConfig | None = None,
    window: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
    visual: dict[str, Any] | None = None,
    telemetry_type: str = "",
    sort_key: str = "",
    hilight_tags: Sequence[float] | None = None,
    hilight_candidates: Sequence[Sequence[float]] | None = None,
    bitrate_density: float | None = None,
    is_high_fps: bool = False,
    domain_prior: dict[str, Any] | None = None,
    vlm_call: bool = False,
) -> PrescreenRecord:
    """Compose one auditable screening record for a candidate window.

    Telemetry signals win when present; visual metrics from sampled frames fill the
    exposure and uniformity gaps so the dark/featureless gates still evaluate on
    clips without a sensor track.
    """
    cfg = config or PrescreenConfig()
    window = window or aggregate_window(None, start, end)
    visual = visual or {}
    audio = audio or {}

    signals: dict[str, Any] = {
        "yavg": window.get("yavg"),
        "unif": window.get("unif"),
        "roll_deg": window.get("roll_deg"),
        "pitch_deg": window.get("pitch_deg"),
        "tilt_deg": window.get("tilt_deg"),
        "gyro_rms": window.get("gyro_rms"),
        "gps_speed_ms": window.get("gps_speed_ms"),
        "face_detected": window.get("face_detected"),
        "laplacian_sharpness": visual.get("laplacian_sharpness"),
        "hue_changes_per_s": visual.get("hue_changes_per_s"),
    }
    fallback_sources: list[str] = []
    if signals["yavg"] is None and visual.get("mean_luma") is not None:
        signals["yavg"] = visual["mean_luma"]
        fallback_sources.append("yavg")
    if signals["unif"] is None and visual.get("luma_uniformity") is not None:
        signals["unif"] = visual["luma_uniformity"]
        fallback_sources.append("unif")

    gates = build_gates(signals, cfg)
    action, reasons = resolve_action(gates, cfg.gate_mode)
    score_info = compute_telemetry_score(signals, cfg.telemetry_weights())

    prior = domain_prior or {}
    if fallback_sources:
        reasons.append(f"{'/'.join(fallback_sources)} from sampled frames (no sensor track)")
    if not gates:
        reasons.append("no gate available")
    elif all(gate.outcome == "pass" for gate in gates):
        reasons.append("all gates within thresholds")

    return PrescreenRecord(
        clip_id=clip_id,
        source_file=source_file,
        sort_key=sort_key,
        candidate_index=candidate_index,
        interval=PrescreenInterval(
            start=round(start, 3),
            end=round(end, 3),
            sample_count=_as_int(window.get("sample_count"), 0),
        ),
        gate_mode="enabled" if cfg.gate_mode == "enabled" else "advisory",
        is_high_fps=is_high_fps,
        hilight_tags=[round(float(tag), 3) for tag in (hilight_tags or [])],
        hilight_candidates=[
            [round(float(pair[0]), 3), round(float(pair[1]), 3)]
            for pair in (hilight_candidates or [])
        ],
        bitrate_density=bitrate_density,
        telemetry_type=telemetry_type,
        scen_dominant=window.get("scen_dominant"),
        scen_confidence=window.get("scen_confidence"),
        scen_coverage=prior.get("coverage"),
        domain_prior_applied=bool(prior.get("fast_path")),
        domain_prior_reason=str(prior.get("fallback_reason", "")),
        yavg=signals.get("yavg"),
        unif=signals.get("unif"),
        roll_deg=window.get("roll_deg"),
        pitch_deg=window.get("pitch_deg"),
        tilt_deg=window.get("tilt_deg"),
        gyro_rms=window.get("gyro_rms"),
        gps_speed_ms=window.get("gps_speed_ms"),
        face_detected=window.get("face_detected"),
        audio_peak_db=audio.get("audio_peak_db"),
        audio_spike_triggered=bool(audio.get("audio_spike_triggered", False)),
        laplacian_sharpness=visual.get("laplacian_sharpness"),
        hsv_entropy=visual.get("hsv_entropy"),
        thresholds={key: float(value) for key, value in cfg.flat_thresholds().items()},
        gates=gates,
        telemetry_score=score_info.get("score"),
        score_components=score_components(score_info, signals),
        action=action,
        vlm_call=vlm_call,
        reason="; ".join(reasons),
    )


@dataclass
class PrescreenResult:
    """Screening output for one clip, shared by the trace and the scan loop."""

    records: list[PrescreenRecord] = field(default_factory=list)
    rejected_intervals: list[tuple[float, float]] = field(default_factory=list)
    deferred_intervals: list[tuple[float, float]] = field(default_factory=list)
    visual: dict[str, Any] = field(default_factory=dict)
    audio_peak_count: int = 0
    duration: float = 0.0

    def action_at(self, timestamp: float) -> tuple[str, PrescreenRecord | None]:
        """Return the screening action that covers ``timestamp``."""
        for record in self.records:
            if record.interval.start <= timestamp <= record.interval.end:
                return record.action, record
        return "keep", None

    def intervals_for(self, action: str) -> list[tuple[float, float]]:
        """Return the intervals that resolved to ``action``."""
        return [
            (record.interval.start, record.interval.end)
            for record in self.records
            if record.action == action
        ]


def screen_clip(
    video_path: str | Path,
    *,
    clip_id: str,
    source_file: str = "",
    sort_key: str = "",
    config: PrescreenConfig | None = None,
    track: TelemetryTrack | None = None,
    telemetry_type: str = "",
    hilight_tags: Sequence[float] | None = None,
    intervals: Sequence[tuple[float, float]] | None = None,
    duration: float | None = None,
    bitrate_density: float | None = None,
    is_high_fps: bool = False,
    has_audio: bool = True,
) -> PrescreenResult:
    """Screen one clip and return per-window records plus the intervals to skip.

    A single FFmpeg pass samples the frames used for both the visual metrics and
    the hue histograms, and a second pass reads audio peak levels. Frames whose
    timestamp falls inside a window are the only ones that contribute to it, so
    the metrics stay aligned with the frames the VLM would have seen. The audio
    pass is skipped when the clip is known to have no audio track.
    """
    cfg = config or PrescreenConfig()
    clip_duration = duration if duration is not None else (probe_duration(video_path) or 0.0)
    result = PrescreenResult(duration=round(float(clip_duration), 3))
    if not cfg.enabled:
        return result

    tags = [float(tag) for tag in (hilight_tags or [])]
    windows: list[tuple[float, float]] = []
    for start, end in intervals or candidate_windows(tags, clip_duration):
        if end > start:
            windows.append((round(float(start), 3), round(float(end), 3)))
    if not windows and clip_duration > 0:
        windows.append((0.0, round(float(clip_duration), 3)))

    frames = extract_raw_frames(
        video_path,
        fps=cfg.sample_fps,
        size=cfg.sample_size,
        keyframes_only=cfg.keyframes_only,
    )
    timestamps = [round(index / cfg.sample_fps, 3) for index in range(len(frames))]
    audio_peaks = extract_audio_peak_levels(video_path) if (cfg.audio_enabled and has_audio) else []
    result.audio_peak_count = len(audio_peaks)

    overall_visual = visual_metrics(frames, timestamps, size=cfg.sample_size)
    result.visual = {key: value for key, value in overall_visual.items() if key != "histograms"}
    prior = telemetry_domain_prior(track, cfg.domain_prior_thresholds, clip_duration)

    for index, (start, end) in enumerate(windows):
        window_frames = [
            frame for frame, stamp in zip(frames, timestamps, strict=False) if start <= stamp <= end
        ]
        window_stamps = [stamp for stamp in timestamps if start <= stamp <= end]
        window_visual = (
            visual_metrics(window_frames, window_stamps, size=cfg.sample_size)
            if window_frames
            else overall_visual
        )
        record = build_prescreen_record(
            clip_id=clip_id,
            source_file=source_file,
            candidate_index=index,
            start=start,
            end=end,
            config=cfg,
            window=aggregate_window(track, start, end),
            audio=audio_window_metrics(
                audio_peaks,
                start,
                end,
                spike_db=cfg.audio_spike_db,
                floor_db=cfg.audio_spike_floor_db,
                window=cfg.audio_window,
            ),
            visual=window_visual,
            telemetry_type=telemetry_type,
            sort_key=sort_key,
            hilight_tags=tags,
            hilight_candidates=windows,
            bitrate_density=bitrate_density,
            is_high_fps=is_high_fps,
            domain_prior=prior,
        )
        result.records.append(record)

    result.rejected_intervals = result.intervals_for("reject")
    result.deferred_intervals = result.intervals_for("defer")
    logger.debug(
        "Prescreen %s: %d window(s), %d reject, %d defer",
        clip_id,
        len(result.records),
        len(result.rejected_intervals),
        len(result.deferred_intervals),
    )
    return result
